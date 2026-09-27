import re

from datasets import load_dataset

def clean_documentation(text):
    # Удаляем маркеры регионов
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

    # Удаляем ссылки
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Удаляем JSDoc-теги и всё после первого тега
    text = re.sub(
        r"(?:^|\s)@\w+\b.*$",
        " ",
        text
    )

    # Нормализуем пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # Убираем оставшуюся пунктуацию в конце
    text = text.rstrip(" :;,")

    return text

def is_good_example(example):
    query = clean_documentation(
        example["func_documentation_string"]
    )

    code = example["func_code_string"].strip()

    # Слишком короткое описание обычно мало что сообщает
    has_enough_words = len(query.split()) >= 4

    return bool(code) and has_enough_words

dataset = load_dataset(
    "code-search-net/code_search_net",
    "javascript",
    split="train",
    streaming=True
)

def load_pairs(split, limit):
    dataset = load_dataset(
        "code-search-net/code_search_net",
        "javascript",
        split=split,
        streaming=True
    )

    # При streaming перемешивание происходит внутри буфера,
    # а не сразу по всему датасету
    dataset = dataset.shuffle(
        seed=42,
        buffer_size=1000
    )

    pairs = []

    for example in dataset:
        if not is_good_example(example):
            continue

        # TODO: добавь словарь с полями:
        # query, code, name, repository
        pairs.append({
            "query": clean_documentation(example["func_documentation_string"]),
            "code": example["func_code_string"].strip(),
            "name": example["func_name"],
            "repository": example["repository_name"]
        })

        # TODO: остановись, когда собрано limit примеров
        if len(pairs) >= limit:
            break

    return pairs
