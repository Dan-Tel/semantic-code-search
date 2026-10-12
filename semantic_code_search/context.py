from pathlib import Path


def build_context(results, root=None):
    """Format retrieved code and locations into one string for a future LLM call."""
    blocks = []
    for number, result in enumerate(results, start=1):
        path = Path(result["path"])
        if root is not None:
            try:
                path = path.resolve().relative_to(Path(root).resolve())
            except ValueError:
                pass

        blocks.append(
            f"[{number}]\n"
            f"Function: {result['name']}\n"
            f"File: {path.as_posix()}\n"
            f"Lines: {result['start_line']}-{result['end_line']}\n"
            f"Code:\n{result['code']}"
        )
    return "\n\n---\n\n".join(blocks)
