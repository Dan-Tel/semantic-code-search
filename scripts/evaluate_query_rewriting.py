import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_ollama import ChatOllama

from semantic_code_search.search import load_repository_search


ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "qwen3:8b"
TOP_K = 3

# Не меняем вопросы во время сравнения.
# Имена функций в нашем sample_repository уникальны.
EVALUATION_CASES = [
    {
        "query": (
            "Where is the authentication token added, "
            "and how does it work?"
        ),
        "expected": "addAuthToken",
    },
    {
        "query": (
            "Where is the authentication token removed "
            "from a request?"
        ),
        "expected": "removeAuthToken",
    },
    {
        "query": (
            "How does the code combine a base URL "
            "with an API endpoint?"
        ),
        "expected": "createApiUrl",
    },
    {
        "query": (
            "Where is the viewport position and zoom "
            "saved in the browser URL?"
        ),
        "expected": "saveViewportToUrl",
    },
    {
        "query": (
            "How does the code prevent zooming "
            "beyond the allowed limits?"
        ),
        "expected": "limitZoom",
    },
    {
        "query": "Where are users sorted alphabetically by name?",
        "expected": "sortUsersByName",
    },
    {
        "query": "Where are deleted users filtered out?",
        "expected": "filterDeletedUsers",
    },
    {
        "query": "How can a user be found by their identifier?",
        "expected": "findUserById",
    },
]


def rewrite_query(llm, query):
    # Та же инструкция и примеры, что в нашем графе.
    response = llm.invoke([
        (
            "system",
            "Convert the question into a short English description "
            "of the code operation to search for. "
            "Start with an action verb. "
            "Keep important technical terms and the intended action. "
            "Remove phrases such as 'where', 'in the code', "
            "'how does it work', and requests for explanations. "
            "Do not invent implementation details or function names. "
            "Output only the search phrase.",
        ),
        (
            "human",
            "Where are expired cache entries cleaned up, "
            "and how does that work?",
        ),
        ("ai", "remove expired cache entries"),
        (
            "human",
            "How does the application read settings from a JSON file?",
        ),
        ("ai", "read application settings from JSON file"),
        ("human", query),
    ])

    rewritten = response.content.strip()
    return rewritten if rewritten else query


def get_rank(results, expected):
    """Позиция правильной функции, начиная с 1, либо None."""
    for rank, result in enumerate(results, start=1):
        if result["name"] == expected:
            return rank

    return None


def calculate_metrics(ranks):
    count = len(ranks)

    return {
        "top1_accuracy": sum(rank == 1 for rank in ranks) / count,
        "recall_at_3": sum(
            rank is not None for rank in ranks
        ) / count,
        "mrr_at_3": sum(
            1 / rank for rank in ranks if rank is not None
        ) / count,
    }


def summarize_results(results):
    return [
        {
            "name": result["name"],
            "score": result["score"],
        }
        for result in results
    ]


def main():
    search, checkpoint = load_repository_search(ROOT)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url="http://localhost:11434",
        temperature=0,
        reasoning=False,
    )

    original_ranks = []
    rewritten_ranks = []
    case_results = []

    improved = 0
    worsened = 0
    unchanged = 0

    for index, case in enumerate(EVALUATION_CASES, start=1):
        query = case["query"]
        expected = case["expected"]

        # 1. Ищем по исходному вопросу.
        original_results = search.retrieve(
            query,
            top_k=TOP_K,
        )

        # 2. Переформулируем и ищем той же моделью.
        rewritten_query = rewrite_query(llm, query)

        rewritten_results = search.retrieve(
            rewritten_query,
            top_k=TOP_K,
        )

        # 3. Находим позицию ожидаемой функции в top-3.
        original_rank = get_rank(original_results, expected)
        rewritten_rank = get_rank(rewritten_results, expected)

        original_ranks.append(original_rank)
        rewritten_ranks.append(rewritten_rank)

        # Сравниваем по reciprocal rank в пределах top-3.
        original_rr = (
            1 / original_rank if original_rank is not None else 0
        )
        rewritten_rr = (
            1 / rewritten_rank if rewritten_rank is not None else 0
        )

        if rewritten_rr > original_rr:
            outcome = "improved"
            improved += 1
        elif rewritten_rr < original_rr:
            outcome = "worsened"
            worsened += 1
        else:
            outcome = "unchanged"
            unchanged += 1

        original_names = [
            result["name"] for result in original_results
        ]
        rewritten_names = [
            result["name"] for result in rewritten_results
        ]

        print(f"\n[{index}/{len(EVALUATION_CASES)}]")
        print("Question:", query)
        print("Rewritten:", rewritten_query)
        print("Expected:", expected)

        print(
            f"Original:  rank={original_rank}, "
            f"results={original_names}"
        )
        print(
            f"Rewritten: rank={rewritten_rank}, "
            f"results={rewritten_names}"
        )
        print("Outcome:", outcome)

        case_results.append({
            "query": query,
            "rewritten_query": rewritten_query,
            "expected": expected,
            "original_rank": original_rank,
            "rewritten_rank": rewritten_rank,
            "outcome": outcome,
            "original_results": summarize_results(original_results),
            "rewritten_results": summarize_results(rewritten_results),
        })

    metrics = {
        "Neural": calculate_metrics(original_ranks),
        "Neural + rewrite": calculate_metrics(rewritten_ranks),
    }

    print("\nMethod           | Top-1  | Recall@3 | MRR@3")
    print("-" * 51)

    for method, values in metrics.items():
        print(
            f"{method:<16} | "
            f"{values['top1_accuracy']:6.1%} | "
            f"{values['recall_at_3']:8.1%} | "
            f"{values['mrr_at_3']:.4f}"
        )

    print(f"\nImproved:  {improved}")
    print(f"Worsened:  {worsened}")
    print(f"Unchanged: {unchanged}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "rewrite_model": MODEL_NAME,
        "temperature": 0,
        "reasoning": False,
        "device": str(search.device),
        "top_k": TOP_K,
        "number_of_functions": len(search.functions),
        "number_of_queries": len(EVALUATION_CASES),
        "metrics": metrics,
        "comparison": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
        },
        "results": case_results,
    }

    # Каждый запуск сохраняется в отдельный файл.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    report_path = (
        ROOT / "reports" / f"query_rewriting_{timestamp}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    main()