import json
from pathlib import Path

from semantic_code_search.augmentation import QueryAugmentationDataset


ROOT = Path(__file__).resolve().parents[1]


def main():
    pairs = json.loads(
        (
            ROOT
            / "artifacts/experiment_data/train_30000_seed42.json"
        ).read_text(encoding="utf-8")
    )

    dataset = QueryAugmentationDataset(
        pairs=pairs,
        paraphrases_path=ROOT / "artifacts/query_paraphrases.json",
    )

    index = 24132

    print("Examples:", len(dataset))
    print("Examples with paraphrases:", len(dataset.variants))
    print("Original:", pairs[index]["query"])

    for attempt in range(5):
        example = dataset[index]

        assert example["code"] == pairs[index]["code"]

        print(f"{attempt + 1}. {example['query']}")


if __name__ == "__main__":
    main()