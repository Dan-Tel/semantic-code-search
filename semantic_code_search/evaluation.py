import torch

from .tokenization import tokenize_batch


def evaluate_retrieval(
    model,
    data_loader,
    tokenizer,
    device,
    query_length,
    code_length
):
    model.eval()

    query_batches = []
    code_batches = []

    with torch.no_grad():
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

            query_batches.append(
                query_vectors.cpu()
            )

            code_batches.append(
                code_vectors.cpu()
            )

    if not query_batches:
        raise ValueError("Evaluation dataset is empty")

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
    number_of_codes = all_code_vectors.size(0)

    if number_of_queries != number_of_codes:
        raise ValueError(
            "Evaluation requires the same number "
            "of queries and code snippets"
        )

    expected_indices = torch.arange(
        number_of_queries
    )

    predicted_indices = similarity_matrix.argmax(
        dim=1
    )

    correct = (
        predicted_indices == expected_indices
    ).sum().item()

    top1_accuracy = correct / number_of_queries

    top_k = min(5, number_of_codes)

    top_indices = similarity_matrix.topk(
        k=top_k,
        dim=1
    ).indices

    hits_at_5 = (
        top_indices
        == expected_indices.unsqueeze(1)
    ).any(dim=1)

    recall_at_5 = (
        hits_at_5.float().mean().item()
    )

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