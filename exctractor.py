from pathlib import Path

SUPPORTED_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}

def find_source_files(repository_path):
    repository = Path(repository_path)
    files = []

    for path in repository.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return files

def load_source_files(paths):
    source_files = []

    for path in paths:
        code = path.read_text(encoding="utf-8")

        source_files.append({
            "path": str(path),
            "code": code
        })

    return source_files

# files = find_source_files("sample_repository")
# source_files = load_source_files(files)

# for source_file in source_files:
#     print("=" * 50)
#     print(source_file["path"])
#     print(source_file["code"])