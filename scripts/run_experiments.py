import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_code_search.augmentation import QueryAugmentationDataset
from semantic_code_search.checkpoint import save_checkpoint
from semantic_code_search.data import load_pairs
from semantic_code_search.device import get_device
from semantic_code_search.evaluation import evaluate_retrieval
from semantic_code_search.model import create_model
from semantic_code_search.tokenization import load_tokenizer
from semantic_code_search.training import train_one_epoch

from scripts.hard_negative_sampler import HardNegativeBatchSampler


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "artifacts/tokenizer.json"
PARAPHRASES_PATH = ROOT / "artifacts/query_paraphrases.json"
DATA_DIR = ROOT / "artifacts/experiment_data"
REPORTS_DIR = ROOT / "reports/experiments"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a code search model from scratch."
    )
    parser.add_argument("--train-limits", type=int, nargs="+", default=[30000])
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--code-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--hard-negatives", action="store_true")
    parser.add_argument("--paraphrases-path", type=Path, default=PARAPHRASES_PATH)
    args = parser.parse_args()

    if min(args.train_limits) < 2 or args.batch_size < 2:
        parser.error("Training requires at least 2 pairs and batch size >= 2.")
    if len(set(args.train_limits)) != len(args.train_limits):
        parser.error("Provide each training size only once.")
    if args.validation_limit < 5:
        parser.error("Recall@5 requires at least 5 validation pairs.")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if not (args.learning_rate > 0 and args.temperature > 0):
        parser.error("Learning rate and temperature must be positive.")
    if not all(3 <= n <= 256 for n in (args.query_length, args.code_length)):
        parser.error("Sequence lengths must be between 3 and 256.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1.")

    return args


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(data):
    text = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cached_pairs(split, limit):
    # Выборка фиксирована. --seed меняет обучение, а не состав validation.
    path = DATA_DIR / f"{split}_{limit}_seed42.json"

    if path.exists():
        print("Loading cached data:", path.name, flush=True)
        pairs = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"Downloading {split}: {limit} pairs...", flush=True)
        pairs = load_pairs(split=split, limit=limit)
        if len(pairs) == limit:
            write_json(path, pairs)

    if len(pairs) != limit:
        raise ValueError(f"Expected {limit} {split} pairs, got {len(pairs)}.")

    return pairs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_experiment(args, train_pairs, validation_loader, tokenizer,
                   device, experiment_dir, common_config):
    # Сбрасываем seed перед созданием каждой модели:
    # эксперименты разного размера стартуют с одинаковых весов.
    set_seed(args.seed)
    model = create_model(tokenizer).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    train_dataset = train_pairs
    augmented_examples = 0

    if args.augment:
        train_dataset = QueryAugmentationDataset(
            pairs=train_pairs,
            paraphrases_path=args.paraphrases_path,
            seed=args.seed,
        )
        augmented_examples = len(train_dataset.variants)

    if args.hard_negatives:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=HardNegativeBatchSampler(
                train_pairs, batch_size=args.batch_size, seed=args.seed
            ),
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=0,
        )

    checkpoint_path = experiment_dir / "best.pt"
    config = {
        **common_config,
        "train_pairs": len(train_pairs),
        "train_sha256": fingerprint(train_pairs),
        "batches_per_epoch": len(train_loader),
        "augmentation": "query_paraphrases" if args.augment else "none",
        "augmented_examples": augmented_examples,
        "paraphrases_sha256": (
            hashlib.sha256(args.paraphrases_path.read_bytes()).hexdigest()
            if args.augment else None
        ),
        "hard_negatives": args.hard_negatives
    }
    write_json(experiment_dir / "config.json", config)

    print(
        f"\nTraining pairs: {len(train_pairs)} | "
        f"Batches: {len(train_loader)} | "
        f"Augmented examples: {augmented_examples}",
        flush=True,
    )

    history = []
    best_row = None

    for epoch in range(1, args.epochs + 1):
        synchronize(device)
        started = perf_counter()

        loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            query_length=args.query_length,
            code_length=args.code_length,
            temperature=args.temperature,
        )

        synchronize(device)
        train_seconds = perf_counter() - started
        started = perf_counter()

        metrics = evaluate_retrieval(
            model=model,
            data_loader=validation_loader,
            tokenizer=tokenizer,
            device=device,
            query_length=args.query_length,
            code_length=args.code_length,
        )

        synchronize(device)
        row = {
            "epoch": epoch,
            "train_loss": loss,
            **metrics,
            "train_seconds": train_seconds,
            "validation_seconds": perf_counter() - started,
            "optimizer_steps": epoch * len(train_loader),
        }
        history.append(row)

        if best_row is None or row["mrr"] > best_row["mrr"]:
            best_row = row
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )

        write_json(experiment_dir / "history.json", history)
        write_csv(experiment_dir / "history.csv", history)

        print(
            f"Epoch {epoch}/{args.epochs} | loss: {loss:.4f} | "
            f"Top-1: {row['top1_accuracy']:.2%} | "
            f"Recall@5: {row['recall_at_5']:.2%} | "
            f"MRR: {row['mrr']:.4f}",
            flush=True,
        )

    summary = {
        "train_pairs": len(train_pairs),
        "best_epoch": best_row["epoch"],
        "top1_accuracy": best_row["top1_accuracy"],
        "recall_at_5": best_row["recall_at_5"],
        "mrr": best_row["mrr"],
        "total_train_minutes": sum(r["train_seconds"] for r in history) / 60,
        "total_optimizer_steps": history[-1]["optimizer_steps"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    write_json(experiment_dir / "summary.json", summary)
    return summary


def main():
    args = parse_args()
    device = get_device()
    print("Device:", device, flush=True)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    all_train_pairs = load_cached_pairs("train", max(args.train_limits))
    validation_pairs = load_cached_pairs("validation", args.validation_limit)
    validation_loader = DataLoader(
        validation_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = REPORTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    common_config = {
        "seed": args.seed,
        "data_seed": 42,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "temperature": args.temperature,
        "query_length": args.query_length,
        "code_length": args.code_length,
        "validation_pairs": len(validation_pairs),
        "validation_sha256": fingerprint(validation_pairs),
        "tokenizer_sha256": hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
        "device": str(device),
        "torch_version": str(torch.__version__),
    }

    summaries = []
    for limit in args.train_limits:
        summary = run_experiment(
            args=args,
            train_pairs=all_train_pairs[:limit],
            validation_loader=validation_loader,
            tokenizer=tokenizer,
            device=device,
            experiment_dir=run_dir / f"train_{limit}",
            common_config=common_config,
        )
        summaries.append(summary)
        write_json(run_dir / "summary.json", summaries)
        write_csv(run_dir / "summary.csv", summaries)

    print("\nPairs | Best epoch | Top-1 | Recall@5 | MRR | Train min")
    for row in summaries:
        print(
            f"{row['train_pairs']:5} | {row['best_epoch']:10} | "
            f"{row['top1_accuracy']:.2%} | {row['recall_at_5']:.2%} | "
            f"{row['mrr']:.4f} | {row['total_train_minutes']:.1f}"
        )
    print("\nResults:", run_dir)


if __name__ == "__main__":
    main()