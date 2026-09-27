from semantic_code_search.baselines.tfidf import (
    TfidfCodeSearch,
)

EVALUATION_CASES = [
    {
        "query": "attach authentication token to request",
        "expected": "addAuthToken",
    },
    {
        "query": "delete credentials from request headers",
        "expected": "removeAuthToken",
    },
    {
        "query": "combine server address with API path",
        "expected": "createApiUrl",
    },
    {
        "query": "order people alphabetically",
        "expected": "sortUsersByName",
    },
    {
        "query": "exclude removed accounts",
        "expected": "filterDeletedUsers",
    },
    {
        "query": "locate account using its identifier",
        "expected": "findUserById",
    },
    {
        "query": "synchronize graph position with browser address",
        "expected": "saveViewportToUrl",
    },
    {
        "query": "keep magnification within allowed range",
        "expected": "limitZoom",
    },
]

def main():
    search_engine = TfidfCodeSearch(
        "tests/fixtures/sample_repository"
    )

    correct_at_1 = 0
    recall_hits = 0
    reciprocal_rank_sum = 0

    for case in EVALUATION_CASES:
        results = search_engine.retrieve(
            case["query"],
            top_k=3
        )

        retrieved_names = [
            result["name"]
            for result in results
        ]

        expected = case["expected"]

        if (
            retrieved_names
            and retrieved_names[0] == expected
        ):
            correct_at_1 += 1

        if expected in retrieved_names:
            recall_hits += 1

            rank = (
                retrieved_names.index(expected) + 1
            )

            reciprocal_rank_sum += 1 / rank
        else:
            rank = None

        print(
            f"expected={expected:<20} "
            f"rank={rank} "
            f"retrieved={retrieved_names}"
        )

    number_of_cases = len(EVALUATION_CASES)

    recall_at_1 = correct_at_1 / number_of_cases
    recall_at_3 = recall_hits / number_of_cases
    mrr = reciprocal_rank_sum / number_of_cases

    print(f"\nRecall@1: {recall_at_1:.2%}")
    print(f"Recall@3: {recall_at_3:.2%}")
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    main()