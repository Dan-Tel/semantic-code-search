from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from data_utils import load_pairs
from small_code_encoder import SmallCodeEncoder


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def tokenize_batch(
    tokenizer,
    texts,
    max_length,
    device
):
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        key: tensor.to(device)
        for key, tensor in inputs.items()
    }


def train_one_epoch(
    model,
    train_loader,
    tokenizer,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in train_loader:
        query_inputs = tokenize_batch(
            tokenizer,
            batch["query"],
            QUERY_LENGTH,
            device
        )

        code_inputs = tokenize_batch(
            tokenizer,
            batch["code"],
            CODE_LENGTH,
            device
        )

        query_vectors = model(
            query_inputs["input_ids"],
            query_inputs["attention_mask"]
        )

        code_vectors = model(
            code_inputs["input_ids"],
            code_inputs["attention_mask"]
        )

        similarity_matrix = (
            query_vectors @ code_vectors.T
        )

        logits = similarity_matrix / TEMPERATURE

        batch_size = logits.size(0)

        labels = torch.arange(
            batch_size,
            device=device
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(
    model,
    validation_loader,
    tokenizer,
    device
):
    model.eval()

    query_batches = []
    code_batches = []

    with torch.no_grad():
        for batch in validation_loader:
            query_inputs = tokenize_batch(
                tokenizer,
                batch["query"],
                QUERY_LENGTH,
                device
            )

            code_inputs = tokenize_batch(
                tokenizer,
                batch["code"],
                CODE_LENGTH,
                device
            )

            query_vectors = model(
                query_inputs["input_ids"],
                query_inputs["attention_mask"]
            )

            code_vectors = model(
                code_inputs["input_ids"],
                code_inputs["attention_mask"]
            )

            query_batches.append(
                query_vectors.cpu()
            )

            code_batches.append(
                code_vectors.cpu()
            )

    all_query_vectors = torch.cat(
        query_batches,
        dim=0
    )

    all_code_vectors = torch.cat(
        code_batches,
        dim=0
    )

    similarity_matrix = (
        all_query_vectors @ all_code_vectors.T
    )

    number_of_queries = all_query_vectors.size(0)

    expected_indices = torch.arange(
        number_of_queries
    )

    # Top-1 accuracy
    predicted_indices = similarity_matrix.argmax(
        dim=1
    )

    correct = (
        predicted_indices == expected_indices
    ).sum().item()

    top1_accuracy = correct / number_of_queries

    # Recall@5
    top5_indices = similarity_matrix.topk(
        k=5,
        dim=1
    ).indices

    hits_at_5 = (
        top5_indices
        == expected_indices.unsqueeze(1)
    ).any(dim=1)

    recall_at_5 = (
        hits_at_5.float().mean().item()
    )

    # Mean Reciprocal Rank
    sorted_indices = similarity_matrix.argsort(
        dim=1,
        descending=True
    )

    matches = (
        sorted_indices
        == expected_indices.unsqueeze(1)
    )

    ranks = (
        matches.float().argmax(dim=1) + 1
    )

    mrr = (
        1.0 / ranks.float()
    ).mean().item()

    return {
        "correct": correct,
        "top1_accuracy": top1_accuracy,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
    }


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

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH),
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )

    model = SmallCodeEncoder(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_mrr = float("-inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            tokenizer,
            optimizer,
            device
        )

        metrics = evaluate(
            model,
            validation_loader,
            tokenizer,
            device
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

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                CHECKPOINT_PATH
            )

            print("New best model saved")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics = evaluate(
        model,
        validation_loader,
        tokenizer,
        device
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