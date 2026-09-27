import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..parsing.typescript import extract_repository_functions


def normalize_text(text):
    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text
    )

    text = text.replace("_", " ")

    return text.lower()


class TfidfCodeSearch:
    def __init__(self, repository_path):
        self.functions = extract_repository_functions(
            repository_path
        )

        if not self.functions:
            raise ValueError(
                f"No functions found in {repository_path}"
            )

        documents = [
            normalize_text(function["code"])
            for function in self.functions
        ]

        self.vectorizer = TfidfVectorizer()

        self.document_vectors = (
            self.vectorizer.fit_transform(documents)
        )

    def retrieve(self, query, top_k=3):
        if top_k <= 0:
            return []

        query_vector = self.vectorizer.transform([
            normalize_text(query)
        ])

        similarities = cosine_similarity(
            query_vector,
            self.document_vectors
        )[0]

        number_of_results = min(
            top_k,
            len(self.functions)
        )

        sorted_indices = similarities.argsort()[
            ::-1
        ][:number_of_results]

        results = []

        for index in sorted_indices:
            score = similarities[index]

            if score <= 0:
                continue

            results.append({
                **self.functions[index],
                "score": float(score),
            })

        return results


def print_results(results):
    for position, result in enumerate(
        results,
        start=1
    ):
        print(
            f"{position}. {result['name']} - "
            f"{result['score']:.4f}"
        )

        print(
            f"   {result['path']}:"
            f"{result['start_line']}-"
            f"{result['end_line']}"
        )