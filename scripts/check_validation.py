import json
from pathlib import Path

from torch.utils.data import DataLoader

from semantic_code_search.checkpoint import load_checkpoint
from semantic_code_search.device import get_device
from semantic_code_search.evaluation import evaluate_retrieval
from semantic_code_search.model import create_model
from semantic_code_search.tokenization import load_tokenizer


def main():
    root = Path(__file__).resolve().parents[1]
    device = get_device()

    tokenizer = load_tokenizer(root / "artifacts/tokenizer.json")
    model = create_model(tokenizer).to(device)

    checkpoint = load_checkpoint(
        root / "checkpoints/small_code_encoder_q2c_best.pt",
        model,
        device,
    )

    validation_path = (
        root / "artifacts/experiment_data/validation_500_seed42.json"
    )
    pairs = json.loads(validation_path.read_text(encoding="utf-8"))

    metrics = evaluate_retrieval(
        model=model,
        data_loader=DataLoader(
            pairs,
            batch_size=1,
            shuffle=False,
        ),
        tokenizer=tokenizer,
        device=device,
        query_length=64,
        code_length=256,
    )

    print("Device:", device)
    print("Epoch:", checkpoint["epoch"])
    print("Saved:", checkpoint["metrics"])
    print("Recomputed:", metrics)


if __name__ == "__main__":
    main()