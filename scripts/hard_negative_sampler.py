import math
import random

from sklearn.feature_extraction.text import TfidfVectorizer


class HardNegativeBatchSampler:
    """Собирает батчи из случайных примеров и их ближайших соседей."""

    def __init__(self, pairs, batch_size, seed, pool_size=1024):
        self.size = len(pairs)
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.rng = random.Random(seed)

        # Соседей ищем по оригинальным train-описаниям, без демо-запросов.
        self.vectors = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        ).fit_transform([pair["query"] for pair in pairs])

        # Не подбираем как сложного соседа точную копию запроса или кода.
        self.queries = [pair["query"].strip().casefold() for pair in pairs]
        self.codes = [pair["code"].strip() for pair in pairs]

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        order = list(range(self.size))
        self.rng.shuffle(order)
        remaining = set(order)

        for start in range(0, self.size, self.pool_size):
            pool = order[start : start + self.pool_size]
            block = self.vectors[pool]
            similarities = (block @ block.T).tocsr()
            local_index = {index: row for row, index in enumerate(pool)}

            batch = []
            for anchor in pool:
                if anchor not in remaining:
                    continue

                remaining.remove(anchor)
                batch.append(anchor)

                if len(batch) < self.batch_size:
                    row = similarities.getrow(local_index[anchor])
                    neighbors = sorted(
                        zip(row.data, row.indices),
                        key=lambda pair: (-pair[0], pair[1]),
                    )
                    for _, column in neighbors:
                        candidate = pool[column]
                        if (
                            candidate in remaining
                            and self.queries[candidate] != self.queries[anchor]
                            and self.codes[candidate] != self.codes[anchor]
                        ):
                            remaining.remove(candidate)
                            batch.append(candidate)
                            break

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            # Переносим неполный батч в следующий блок, чтобы число
            # шагов осталось тем же, что и у обычного DataLoader.
            if batch:
                for candidate in order[start + self.pool_size :]:
                    if candidate in remaining:
                        remaining.remove(candidate)
                        batch.append(candidate)
                        if len(batch) == self.batch_size:
                            break
                yield batch