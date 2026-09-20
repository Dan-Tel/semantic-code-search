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

train_pairs = load_pairs(
    split="train",
    limit=10
)

# print(f"Collected: {len(train_pairs)}")

# for pair in train_pairs[:3]:
#     print("=" * 70)
#     print("Function:", pair["name"])
#     print("Repository:", pair["repository"])
#     print("Query:", pair["query"])
#     print("Code:", pair["code"][:200])

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from search import normalize_text

batch = train_pairs[:3]

queries = [
    pair["query"] for pair in batch
]

codes = [
    pair["code"] for pair in batch
]

batch_vectorizer = TfidfVectorizer()

# Один словарь нужен, чтобы query и code получили
# векторы в одном пространстве
batch_vectorizer.fit(queries + codes)

query_vectors = batch_vectorizer.transform(queries)
code_vectors = batch_vectorizer.transform(codes)

similarity_matrix = cosine_similarity(
    query_vectors,
    code_vectors
)

# print("Similarity matrix:")
# print(similarity_matrix)
# print(np.round(similarity_matrix, 3))

expected_indices = np.arange(len(batch))
predicted_indices = similarity_matrix.argmax(axis=1)

# print("Expected:", expected_indices)
# print("Predicted:", predicted_indices)

temperature = 0.1

# Чем меньше temperature, тем сильнее подчёркиваются
# различия между similarity
logits = similarity_matrix / temperature

# # Вычитаем максимум для численной стабильности
shifted_logits = logits - logits.max(
    axis=1,
    keepdims=True
)

exp_scores = np.exp(shifted_logits)

probabilities = exp_scores / exp_scores.sum(
    axis=1,
    keepdims=True
)

# print("Probabilities:")
# print(np.round(probabilities, 3))

batch_size = len(batch)

correct_probabilities = probabilities[
    np.arange(batch_size),
    expected_indices
]

# print("Correct probabilities:")
# print(np.round(correct_probabilities, 3))

loss = -np.log(correct_probabilities).mean()
# print("Loss:", round(loss, 4))


import torch
import torch.nn.functional as F

logits_tensor = torch.tensor(
    similarity_matrix / temperature,
    dtype=torch.float32
)

print(logits_tensor)

labels = torch.arange(len(batch))

pytorch_loss = F.cross_entropy(
    logits_tensor,
    labels
)

print("Manual loss:", round(loss, 4))
print("PyTorch loss:", round(pytorch_loss.item(), 4))