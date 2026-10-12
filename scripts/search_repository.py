import argparse
from pathlib import Path

from semantic_code_search.context import build_context
from semantic_code_search.search import load_repository_search

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Search TypeScript functions")
    parser.add_argument("--repository", type=Path,
                        default=ROOT / "tests/fixtures/sample_repository")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0:
        parser.error("--top-k and --batch-size must be positive")

    search, _ = load_repository_search(ROOT, args.repository, args.batch_size)
    while True:
        try:
            query = input("\nSearch query (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = search.retrieve(query, top_k=args.top_k)
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['name']} | score: {result['score']:.4f}")
            print(f"   {result['path']}:{result['start_line']}-{result['end_line']}")
        if args.show_context:
            print("\nContext for LLM:\n")
            print(build_context(results, root=ROOT))


if __name__ == "__main__":
    main()
