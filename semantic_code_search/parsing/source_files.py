from pathlib import Path

SUPPORTED_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}

def find_source_files(repository_path):
    repository = Path(repository_path)
    files = []

    for path in repository.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return files
