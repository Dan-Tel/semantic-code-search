import re

from datasets import load_dataset


DATASET_NAME = "code-search-net/code_search_net"
LANGUAGE = "javascript"

SHUFFLE_SEED = 42
SHUFFLE_BUFFER_SIZE = 1000
MIN_QUERY_WORDS = 4


def clean_documentation(text):
    text = text or ""

    text = re.sub(
        r"#endregion\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"#region[^\r\n]*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"(?:^|\s)@\w+\b.*$",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text.rstrip(" :;,")


def create_pair(example):
    query = clean_documentation(
        example["func_documentation_string"]
    )

    code = (
        example["func_code_string"] or ""
    ).strip()

    if not code:
        return None

    if len(query.split()) < MIN_QUERY_WORDS:
        return None

    return {
        "query": query,
        "code": code,
        "name": example["func_name"] or "",
        "repository": example["repository_name"],
    }


def load_pairs(split, limit):
    if limit <= 0:
        return []

    dataset = load_dataset(
        DATASET_NAME,
        LANGUAGE,
        split=split,
        streaming=True
    )

    dataset = dataset.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE
    )

    pairs = []

    for example in dataset:
        pair = create_pair(example)

        if pair is None:
            continue

        pairs.append(pair)

        if len(pairs) >= limit:
            break

    return pairs