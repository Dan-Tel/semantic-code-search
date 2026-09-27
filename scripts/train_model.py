from pathlib import Path

import torch
from torch.utils.data import DataLoader

from semantic_code_search.data import load_pairs
from semantic_code_search.model import create_model
from semantic_code_search.tokenization import (
    load_tokenizer,
    tokenize_batch,
)
from semantic_code_search.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from semantic_code_search.training import (
    train_one_epoch,
)
from semantic_code_search.evaluation import (
    evaluate_retrieval,
)
from semantic_code_search.device import get_device


# Configuration
TRAIN_LIMIT = 5000
VALIDATION_LIMIT = 500

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
TEMPERATURE = 0.1

QUERY_LENGTH = 64
CODE_LENGTH = 256

TOKENIZER_PATH = Path("artifacts/tokenizer.json")
CHECKPOINT_PATH = Path(
    "checkpoints/small_code_encoder_q2c_best.pt"
)


def main():
    torch.manual_seed(42)

    device = get_device()
    print("Device:", device)

    train_pairs = load_pairs(
        split="train",
        limit=TRAIN_LIMIT
    )

    validation_pairs = load_pairs(
        split="validation",
        limit=VALIDATION_LIMIT
    )

    train_loader = DataLoader(
        train_pairs,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_pairs,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Training pairs:", len(train_pairs))
    print("Validation pairs:", len(validation_pairs))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    tokenizer = load_tokenizer(
        TOKENIZER_PATH
    )

    model = create_model(
        tokenizer
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_mrr = float("-inf")

    for epoch in range(NUM_EPOCHS):
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

        metrics = evaluate_retrieval(
            model=model,
            data_loader=validation_loader,
            tokenizer=tokenizer,
            device=device,
            query_length=QUERY_LENGTH,
            code_length=CODE_LENGTH,
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"loss: {train_loss:.4f} | "
            f"Top-1: {metrics['top1_accuracy']:.2%} | "
            f"Recall@5: {metrics['recall_at_5']:.2%} | "
            f"MRR: {metrics['mrr']:.4f}"
        )

        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]

            save_checkpoint(
                checkpoint_path=CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=metrics,
            )

            print("New best model saved")

    checkpoint = load_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        model=model,
        device=device,
    )

    final_metrics = evaluate_retrieval(
        model=model,
        data_loader=validation_loader,
        tokenizer=tokenizer,
        device=device,
        query_length=QUERY_LENGTH,
        code_length=CODE_LENGTH,
    )

    print("\nBest checkpoint")
    print("Epoch:", checkpoint["epoch"])
    print("Correct:", final_metrics["correct"])
    print(
        f"Top-1: "
        f"{final_metrics['top1_accuracy']:.2%}"
    )
    print(
        f"Recall@5: "
        f"{final_metrics['recall_at_5']:.2%}"
    )
    print(
        f"MRR: "
        f"{final_metrics['mrr']:.4f}"
    )


if __name__ == "__main__":
    main()