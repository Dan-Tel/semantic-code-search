import argparse
from pathlib import Path

import torch

from semantic_code_search.checkpoint import load_checkpoint
from semantic_code_search.device import get_device
from semantic_code_search.model import create_model
from semantic_code_search.parsing.typescript import (
    extract_repository_functions,
)
from semantic_code_search.tokenization import (
    load_tokenizer,
    tokenize_batch,
)
from scripts.paraphrase_cases import PARAPHRASE_CASES


ROOT = Path(__file__).resolve().parents[1]

CASES = [
    (
        "addAuthToken",
        "add authentication token to request headers",
        "attach bearer credentials to an outgoing HTTP request",
    ),
    (
        "removeAuthToken",
        "remove authentication token from request headers",
        "strip authorization credentials before sending a request",
    ),
    (
        "createApiUrl",
        "combine a base URL and an endpoint",
        "build a service address by joining its root with a resource path",
    ),
    (
        "saveViewportToUrl",
        "save viewport position and zoom in the URL",
        "persist the current map location and magnification in the browser address",
    ),
    (
        "limitZoom",
        "limit zoom between minimum and maximum",
        "clamp the magnification so it stays within the permitted bounds",
    ),
    (
        "sortUsersByName",
        "sort users alphabetically by name",
        "arrange people in alphabetical order using their names",
    ),
    (
        "filterDeletedUsers",
        "filter out deleted users",
        "keep only people whose records have not been marked as removed",
    ),
    (
        "findUserById",
        "find a user by id",
        "look up a person using their identifier",
    ),
]


def encode(model, tokenizer, texts, max_length, device):
    inputs = tokenize_batch(
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        device=device,
    )

    with torch.no_grad():
        return model(
            inputs["input_ids"],
            inputs["attention_mask"],
        )


def evaluate_cases(
    model,
    tokenizer,
    functions,
    code_vectors,
    cases,
    device,
):
    query_vectors = encode(
        model,
        tokenizer,
        [query for query, expected in cases],
        max_length=64,
        device=device,
    )

    similarities = query_vectors @ code_vectors.T

    rankings = similarities.argsort(
        dim=1,
        descending=True,
    ).cpu().tolist()

    top1_hits = 0
    top3_hits = 0
    reciprocal_rank_sum = 0.0

    for row, (query, expected) in enumerate(cases):
        indices = rankings[row]
        names = [functions[index]["name"] for index in indices]

        rank = names.index(expected) + 1

        top1_hits += int(rank == 1)
        top3_hits += int(rank <= 3)
        reciprocal_rank_sum += 1 / rank

        print(f"\nQuery: {query}")
        print(f"Expected: {expected} | Rank: {rank}")

        for position, index in enumerate(indices[:3], start=1):
            score = similarities[row, index].item()

            print(
                f"  {position}. {functions[index]['name']}"
                f" | {score:.4f}"
            )

    count = len(cases)

    return {
        "top1": top1_hits / count,
        "recall_at_3": top3_hits / count,
        "mrr": reciprocal_rank_sum / count,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to checkpoints to compare",
    )

    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    tokenizer = load_tokenizer(
        ROOT / "artifacts/tokenizer.json"
    )

    functions = extract_repository_functions(
        ROOT / "tests/fixtures/sample_repository"
    )

    if not functions:
        raise ValueError("No functions found")

    names = [function["name"] for function in functions]

    for expected, _, _ in CASES:
        if names.count(expected) != 1:
            raise ValueError(
                f"Expected exactly one function named {expected}"
            )

    query_sets = {
        "direct": [
            (direct, expected)
            for expected, direct, paraphrase in CASES
        ],
        "paraphrased": [
            (paraphrase, expected)
            for expected, direct, paraphrase in CASES
        ],
        "paraphrases_40": PARAPHRASE_CASES,
    }

    summary = []

    for checkpoint_number, value in enumerate(
        args.checkpoints,
        start=1,
    ):
        checkpoint_path = Path(value).expanduser()

        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path

        model = create_model(tokenizer).to(device)

        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
        )

        model.eval()

        label = f"Model {checkpoint_number}"

        print(f"\n{'=' * 70}")
        print(f"{label}: {checkpoint_path}")
        print("Epoch:", checkpoint.get("epoch"))
        print("Saved validation metrics:", checkpoint.get("metrics"))

        code_vectors = encode(
            model,
            tokenizer,
            [function["code"] for function in functions],
            max_length=256,
            device=device,
        )

        for set_name, cases in query_sets.items():
            print(f"\n--- {set_name} ---")

            metrics = evaluate_cases(
                model=model,
                tokenizer=tokenizer,
                functions=functions,
                code_vectors=code_vectors,
                cases=cases,
                device=device,
            )

            summary.append((label, set_name, metrics))

        del code_vectors, model

    print("\nModel   | Query set   | Top-1  | Recall@3 | MRR")

    for label, set_name, metrics in summary:
        print(
            f"{label:<7} | "
            f"{set_name:<11} | "
            f"{metrics['top1']:6.1%} | "
            f"{metrics['recall_at_3']:8.1%} | "
            f"{metrics['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()