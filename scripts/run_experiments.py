import csv
import gc
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_code_search.checkpoint import save_checkpoint
from semantic_code_search.data import load_pairs
from semantic_code_search.device import get_device
from semantic_code_search.evaluation import evaluate_retrieval
from semantic_code_search.model import create_model
from semantic_code_search.tokenization import load_tokenizer
from semantic_code_search.training import train_one_epoch

from semantic_code_search.augmentation import QueryAugmentationDataset


ROOT = Path(__file__).resolve().parents[1]

SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
TEMPERATURE = 0.05
QUERY_LENGTH = 64
CODE_LENGTH = 256

VALIDATION_LIMIT = 500
TRAIN_LIMITS = [10000]

TRAIN_LIMITS = [30000]
EPOCHS = 3
TEMPERATURE = 0.05
BATCH_SIZE = 32

TOKENIZER_PATH = ROOT / "artifacts/tokenizer.json"
DATA_DIR = ROOT / "artifacts/experiment_data"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Сначала записываем временный файл.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_cached_pairs(split, limit):
    path = DATA_DIR / f"{split}_{limit}_seed42.json"

    if path.exists():
        print("Loading cached data:", path.name, flush=True)
        pairs = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"Downloading {split}: {limit} pairs...", flush=True)
        pairs = load_pairs(split=split, limit=limit)

        if len(pairs) != limit:
            raise ValueError(
                f"Expected {limit} {split} pairs, got {len(pairs)}"
            )

        write_json(path, pairs)

    if len(pairs) != limit:
        raise ValueError(f"Unexpected dataset size in {path}")

    return pairs


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def release_memory(device):
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def fingerprint(data):
    text = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_experiment(
    train_pairs,
    validation_pairs,
    tokenizer,
    initial_weights,
    device,
    run_dir,
    common_config,
):
    set_seed()

    train_limit = len(train_pairs)
    experiment_dir = run_dir / f"train_{train_limit}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = experiment_dir / "best.pt"

    generator = torch.Generator().manual_seed(SEED)
    
    train_dataset = QueryAugmentationDataset(
        pairs=train_pairs,
        paraphrases_path=ROOT / "artifacts/query_paraphrases.json",
        seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_pairs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(tokenizer)

    # Одинаковые начальные веса для обоих экспериментов.
    model.load_state_dict(initial_weights)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    config = {
        **common_config,
        "train_pairs": train_limit,
        "train_sha256": fingerprint(train_pairs),
        "batches_per_epoch": len(train_loader),
        "augmentation": "query_paraphrases",
        "augmented_examples": len(train_dataset.variants),
        "paraphrases_sha256": hashlib.sha256(
            (ROOT / "artifacts/query_paraphrases.json").read_bytes()
        ).hexdigest(),
    }
    write_json(experiment_dir / "config.json", config)

    print(
        f"\n=== Training pairs: {train_limit} | "
        f"Batches: {len(train_loader)} ===",
        flush=True,
    )

    history = []
    best_mrr = float("-inf")
    best_row = None

    for epoch in range(1, EPOCHS + 1):
        synchronize(device)
        started_at = perf_counter()

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            query_length=QUERY_LENGTH,
            code_length=CODE_LENGTH,
            temperature=TEMPERATURE,
        )

        synchronize(device)
        train_seconds = perf_counter() - started_at
        validation_started = perf_counter()

        metrics = evaluate_retrieval(
            model=model,
            data_loader=validation_loader,
            tokenizer=tokenizer,
            device=device,
            query_length=QUERY_LENGTH,
            code_length=CODE_LENGTH,
        )

        synchronize(device)
        validation_seconds = perf_counter() - validation_started

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "optimizer_steps": epoch * len(train_loader),
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss: {train_loss:.4f} | "
            f"Top-1: {metrics['top1_accuracy']:.2%} | "
            f"Recall@5: {metrics['recall_at_5']:.2%} | "
            f"MRR: {metrics['mrr']:.4f} | "
            f"train: {train_seconds / 60:.1f} min",
            flush=True,
        )

        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_row = dict(row)

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )
            print("Best checkpoint saved:", checkpoint_path, flush=True)

        # Сохраняем результаты после каждой эпохи.
        write_json(experiment_dir / "history.json", history)

        with (experiment_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "train_pairs": train_limit,
        "best_epoch": best_row["epoch"],
        "top1_accuracy": best_row["top1_accuracy"],
        "recall_at_5": best_row["recall_at_5"],
        "mrr": best_row["mrr"],
        "total_train_minutes": sum(
            row["train_seconds"] for row in history
        ) / 60,
        "total_optimizer_steps": EPOCHS * len(train_loader),
        "checkpoint": str(checkpoint_path),
    }

    write_json(experiment_dir / "summary.json", summary)

    del optimizer, model, train_loader, validation_loader
    release_memory(device)

    return summary


def main():
    device = get_device()
    print("Device:", device, flush=True)

    # Загружаем 10 000 один раз.
    # Первый эксперимент использует первые 5 000 из этой же выборки.
    all_train_pairs = load_cached_pairs(
        "train", max(TRAIN_LIMITS)
    )
    validation_pairs = load_cached_pairs(
        "validation", VALIDATION_LIMIT
    )

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    set_seed()
    initial_model = create_model(tokenizer)

    initial_weights = {
        name: tensor.detach().cpu().clone()
        for name, tensor in initial_model.state_dict().items()
    }
    del initial_model

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_dir = ROOT / "reports/experiments" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    common_config = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "temperature": TEMPERATURE,
        "query_length": QUERY_LENGTH,
        "code_length": CODE_LENGTH,
        "validation_pairs": len(validation_pairs),
        "validation_sha256": fingerprint(validation_pairs),
        "tokenizer_sha256": hashlib.sha256(
            TOKENIZER_PATH.read_bytes()
        ).hexdigest(),
        "device": str(device),
        "torch_version": str(torch.__version__),
    }

    summaries = []

    for train_limit in TRAIN_LIMITS:
        summary = run_experiment(
            train_pairs=all_train_pairs[:train_limit],
            validation_pairs=validation_pairs,
            tokenizer=tokenizer,
            initial_weights=initial_weights,
            device=device,
            run_dir=run_dir,
            common_config=common_config,
        )

        summaries.append(summary)
        write_json(run_dir / "summary.json", summaries)

        with (run_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(summary)
            )
            writer.writeheader()
            writer.writerows(summaries)

    print("\nPairs | Best epoch | Top-1 | Recall@5 | MRR | Train min")

    for row in summaries:
        print(
            f"{row['train_pairs']:5} | "
            f"{row['best_epoch']:10} | "
            f"{row['top1_accuracy']:.2%} | "
            f"{row['recall_at_5']:.2%} | "
            f"{row['mrr']:.4f} | "
            f"{row['total_train_minutes']:.1f}"
        )

    print("\nResults:", run_dir)


if __name__ == "__main__":
    main()