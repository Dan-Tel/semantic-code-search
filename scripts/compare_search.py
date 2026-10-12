import json
from pathlib import Path

from semantic_code_search.baselines.tfidf import TfidfCodeSearch
from semantic_code_search.search import load_repository_search
from scripts.evaluate_tfidf import EVALUATION_CASES

ROOT = Path(__file__).resolve().parents[1]

DIRECT_CASES = [
    ("add an authentication token to request headers", "addAuthToken"),
    ("remove the authentication token from a request", "removeAuthToken"),
    ("combine a base URL and an endpoint", "createApiUrl"),
    ("save the viewport position and zoom in the URL", "saveViewportToUrl"),
    ("keep zoom between minimum and maximum values", "limitZoom"),
    ("sort users alphabetically by name", "sortUsersByName"),
    ("exclude users marked as deleted", "filterDeletedUsers"),
    ("find a user by id", "findUserById"),
]


def evaluate_cases(engines, cases):
    totals = {name: {"top1": 0, "top3": 0, "rr": 0.0} for name in engines}
    query_results = []
    for query, expected in cases:
        rankings = {}
        print(f"\nQuery: {query}\nExpected: {expected}")
        for method, engine in engines.items():
            names = [result["name"] for result in engine.retrieve(query, top_k=3)]
            rankings[method] = names
            rank = names.index(expected) + 1 if expected in names else None
            if rank is not None:
                totals[method]["top1"] += int(rank == 1)
                totals[method]["top3"] += 1
                totals[method]["rr"] += 1 / rank
            print(f"  {method}: rank={rank}, results={names}")
        query_results.append({"query": query, "expected": expected, "rankings": rankings})

    count = len(cases)
    metrics = {
        method: {
            "top1_accuracy": values["top1"] / count,
            "recall_at_3": values["top3"] / count,
            "mrr_at_3": values["rr"] / count,
        }
        for method, values in totals.items()
    }
    return metrics, query_results


def main():
    repository = ROOT / "tests/fixtures/sample_repository"
    neural, checkpoint = load_repository_search(ROOT, repository)
    engines = {"Neural": neural, "TF-IDF": TfidfCodeSearch(repository)}
    query_sets = {
        "direct": DIRECT_CASES,
        "paraphrased": [(case["query"], case["expected"]) for case in EVALUATION_CASES],
    }
    for query_set, cases in query_sets.items():
        print(f"\nQuery set: {query_set}")
        metrics, results = evaluate_cases(engines, cases)
        print("\nMethod   | Top-1  | Recall@3 | MRR@3")
        for method, values in metrics.items():
            print(f"{method:<8} | {values['top1_accuracy']:6.1%} | "
                  f"{values['recall_at_3']:8.1%} | {values['mrr_at_3']:.4f}")
        report = {
            "query_set": query_set,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "device": str(neural.device),
            "number_of_functions": len(neural.functions),
            "number_of_queries": len(cases),
            "metrics": metrics,
            "results": results,
        }
        report_path = ROOT / "reports" / f"{query_set}_results.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Report saved:", report_path)


if __name__ == "__main__":
    main()
