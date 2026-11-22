import json
import random
from pathlib import Path


class QueryAugmentationDataset:
    def __init__(self, pairs, paraphrases_path, seed=42):
        self.pairs = pairs
        self.random = random.Random(seed)
        self.variants = {}

        entries = json.loads(
            Path(paraphrases_path).read_text(encoding="utf-8")
        )

        for entry in entries:
            index = entry["index"]

            if not 0 <= index < len(pairs):
                raise ValueError(f"Invalid example index: {index}")

            if index in self.variants:
                raise ValueError(f"Duplicate example index: {index}")

            original_query = pairs[index]["query"]

            if original_query != entry["expected_query"]:
                raise ValueError(
                    f"Query mismatch at index {index}. "
                    "Paraphrases belong to a different dataset."
                )

            paraphrases = entry["paraphrases"]

            if not paraphrases or any(
                not isinstance(text, str) or not text.strip()
                for text in paraphrases
            ):
                raise ValueError(
                    f"Invalid paraphrases at index {index}"
                )

            self.variants[index] = [
                original_query,
                *paraphrases,
            ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = dict(self.pairs[index])

        if index in self.variants:
            pair["query"] = self.random.choice(
                self.variants[index]
            )

        return pair