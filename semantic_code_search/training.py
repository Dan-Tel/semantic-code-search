import torch
import torch.nn.functional as F

from .tokenization import tokenize_batch


def train_one_epoch(
    model,
    data_loader,
    tokenizer,
    optimizer,
    device,
    query_length,
    code_length,
    temperature
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        query_inputs = tokenize_batch(
            tokenizer,
            batch["query"],
            query_length,
            device
        )

        code_inputs = tokenize_batch(
            tokenizer,
            batch["code"],
            code_length,
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

        logits = similarity_matrix / temperature
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

    if total_examples == 0:
        raise ValueError("Training dataset is empty")

    return total_loss / total_examples