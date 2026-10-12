from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .device import get_device
from .model import create_model
from .parsing.typescript import extract_repository_functions
from .tokenization import load_tokenizer, tokenize_batch


class NeuralCodeSearch:
    def __init__(self, model, tokenizer, functions, device, batch_size=32):
        if not functions:
            raise ValueError("No functions found in the repository")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.functions = list(functions)
        self.device = device

        batches = []
        with torch.no_grad():
            for start in range(0, len(self.functions), batch_size):
                functions_batch = self.functions[start:start + batch_size]
                inputs = tokenize_batch(
                    tokenizer=self.tokenizer,
                    texts=[function["code"] for function in functions_batch],
                    max_length=256,
                    device=self.device,
                )
                vectors = self.model(inputs["input_ids"], inputs["attention_mask"])
                batches.append(vectors)

        self.code_vectors = torch.cat(batches, dim=0)

    def retrieve(self, query, top_k=3):
        if not query.strip() or top_k <= 0:
            return []

        self.model.eval()
        inputs = tokenize_batch(
            tokenizer=self.tokenizer,
            texts=[query],
            max_length=64,
            device=self.device,
        )
        with torch.no_grad():
            query_vector = self.model(inputs["input_ids"], inputs["attention_mask"])
            scores = (query_vector @ self.code_vectors.T)[0]

        indices = scores.argsort(descending=True)[:top_k].tolist()
        return [
            {**self.functions[index], "score": scores[index].item()}
            for index in indices
        ]


def load_repository_search(root, repository_path=None, batch_size=32):
    """Load the saved model and index repository functions once."""
    root = Path(root).resolve()
    repository_path = (
        Path(repository_path).resolve()
        if repository_path is not None
        else root / "tests/fixtures/sample_repository"
    )
    device = get_device()
    tokenizer = load_tokenizer(root / "artifacts/tokenizer.json")
    model = create_model(tokenizer).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path=root / "checkpoints/small_code_encoder_q2c_best.pt",
        model=model,
        device=device,
    )
    functions = extract_repository_functions(repository_path)
    search = NeuralCodeSearch(model, tokenizer, functions, device, batch_size)
    print("Device:", device)
    print("Loaded epoch:", checkpoint.get("epoch"))
    print("Functions found:", len(functions))
    print("Code vectors shape:", search.code_vectors.shape)
    return search, checkpoint
