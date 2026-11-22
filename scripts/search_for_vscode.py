import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    "__pycache__",
    "vendor",
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repository", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ask", action="store_true")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--no-rewrite", action="store_true")

    args = parser.parse_args()

    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    repository = Path(args.repository).resolve()

    if not repository.is_dir():
        raise ValueError(f"Repository not found: {repository}")

    if not args.query.strip() or args.top_k < 1:
        raise ValueError("Query must not be empty; top-k must be positive")

    with redirect_stdout(sys.stderr):
        payload = run_search(args, repository)

    print(json.dumps(payload, ensure_ascii=False, default=str))


def extract_workspace_functions(repository):
    from semantic_code_search.parsing.typescript import extract_functions

    functions = []

    for directory, directories, filenames in os.walk(
        repository,
        followlinks=False,
    ):
        directories[:] = sorted(
            name
            for name in directories
            if name not in SKIP_DIRECTORIES
            and not name.startswith(".")
        )

        for filename in sorted(filenames):
            path = Path(directory) / filename

            if (
                path.suffix != ".ts"
                or filename.endswith(".d.ts")
                or path.is_symlink()
            ):
                continue

            functions.extend(extract_functions(path))

    return functions


def run_search(args, repository):
    from semantic_code_search.search import load_function_search

    functions = extract_workspace_functions(repository)
    if not functions:
        return {
            "results": [],
            "function_count": 0,
            "device": None,
        }

    search, checkpoint = load_function_search(ROOT, functions)

    search_query = args.query
    warnings = []
    llm = None

    if args.ask:
        try:
            from scripts.vscode_answer import (
                answer_question,
                create_llm,
                rewrite_query,
            )

            llm = create_llm(args.ollama_model)

        except Exception as error:
            warnings.append(f"Qwen initialization failed: {error}")

        if llm is not None and not args.no_rewrite:
            try:
                search_query = rewrite_query(llm, args.query)
            except Exception as error:
                warnings.append(
                    f"Using the original question: query rewrite failed. {error}"
                )

    results = search.retrieve(
        search_query,
        top_k=args.top_k,
    )

    payload = {
        "results": results,
        "search_query": search_query,
        "warnings": warnings,
        "function_count": len(functions),
        "device": str(search.device),
        "epoch": checkpoint.get("epoch"),
    }

    if args.ask:
        payload["answer"] = (
            "No answer generated. See warnings and retrieved sources below."
        )

        if llm is not None:
            try:
                payload["answer"] = answer_question(
                    llm,
                    args.query,
                    results,
                    repository,
                )
            except Exception as error:
                warnings.append(
                    f"Answer generation failed. Check Ollama. {error}"
                )

    return payload


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
