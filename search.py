import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from exctractor import find_source_files, load_source_files

from typescript_parser import extract_repository_functions

def normalize_text(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")

    return text.lower()

paths = find_source_files("sample_repository")
source_files = load_source_files(paths)

functions = extract_repository_functions("sample_repository")

documents = [
    normalize_text(function["code"]) for function in functions 
]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)

def search_files(query, top_k=3):
    # преобразовать запрос
    query_vector = vectorizer.transform([normalize_text(query)])

    # посчитать cosine similarity
    similarities = cosine_similarity(query_vector, document_vectors)

    # отсортировать индексы
    scores = similarities[0]
    sorted_indices = scores.argsort()[::-1][:top_k]

    for i, idx in enumerate(sorted_indices):
        # получить source_files[idx]
        data = source_files[idx]

        # вывести path и score
        print(f"{i + 1}. {data['path']} - {scores[idx]:.4f}")

def search_functions(query, top_k=3):
    # 1. Нормализовать и преобразовать запрос
    query_vector = vectorizer.transform([normalize_text(query)])

    # 2. Посчитать cosine similarity
    similarities = cosine_similarity(query_vector, document_vectors)

    # 3. Получить top_k индексов
    scores = similarities[0]
    sorted_indices = scores.argsort()[::-1][:top_k]

    for i, idx in enumerate(sorted_indices):
        # 4. Для каждого индекса взять functions[idx]
        function = functions[idx]

        # 5. Вывести name, path, lines и score
        print(f"{i + 1}. {function['name']} - {scores[idx]:.4f}")
        print(f"   {function['path']}:{function['start_line']}-{function['end_line']}")

def retrieve_functions(query, top_k=3):
    # вычисления остаются прежними
    # 1. Нормализовать и преобразовать запрос
    query_vector = vectorizer.transform([normalize_text(query)])

    # 2. Посчитать cosine similarity
    similarities = cosine_similarity(query_vector, document_vectors)

    # 3. Получить top_k индексов
    scores = similarities[0]
    sorted_indices = scores.argsort()[::-1][:top_k]

    results = []

    for idx in sorted_indices:
        if scores[idx] <= 0:
            continue

        function = functions[idx]

        results.append({
            **function,
            "score": float(scores[idx]),
        })

    return results

def print_results(results):
    for i, result in enumerate(results):
        print(
            f"{i + 1}. {result['name']} - "
            f"{result['score']:.4f}"
        )
        print(
            f"   {result['path']}:"
            f"{result['start_line']}-"
            f"{result['end_line']}"
        )

# results = retrieve_functions(
#     "prevent zoom from exceeding maximum value"
# )

# print_results(results)