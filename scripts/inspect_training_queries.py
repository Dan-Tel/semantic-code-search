import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

path = (
    ROOT
    / "artifacts/experiment_data/train_30000_seed42.json"
)

pairs = json.loads(path.read_text(encoding="utf-8"))

indices = random.Random(42).sample(
    range(len(pairs)),
    k=10,
)

already_augmented = {9012, 24132, 24299, 4572}
candidates = [i for i in range(len(pairs)) if i not in already_augmented]
indices = random.Random(43).sample(candidates, k=30)

for index in indices:
    pair = pairs[index]

    print("=" * 70)
    print("Index:", index)
    print("Function:", pair["name"])
    print("Query:", pair["query"])
    print("Code:")
    print(pair["code"][:1500])