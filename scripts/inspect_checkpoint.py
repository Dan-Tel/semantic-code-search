from pathlib import Path

import torch

from transformers import PreTrainedTokenizerFast

from semantic_code_search.model import SmallCodeEncoder


checkpoint_path = Path(
    "checkpoints/small_code_encoder_q2c_best.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

print("Checkpoint fields:", list(checkpoint.keys()))
print("Epoch:", checkpoint.get("epoch"))
print("Metrics:", checkpoint["metrics"])


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="artifacts/tokenizer.json",
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)

model = SmallCodeEncoder(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    cls_token_id=tokenizer.cls_token_id,
    sep_token_id=tokenizer.sep_token_id,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to("cuda")
model.eval()

print("Model device:", next(model.parameters()).device)


query = "find a user by id"

query_inputs = tokenizer(
    [query],
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
)

query_inputs = {
    key: tensor.to("cuda")
    for key, tensor in query_inputs.items()
}

with torch.no_grad():
    query_vector = model(
        query_inputs["input_ids"],
        query_inputs["attention_mask"],
    )

print("Query:", query)
print("Vector shape:", query_vector.shape)
print("First 10 values:", query_vector[0, :10])


code = """
function findUserById(users, id) {
    return users.find(user => user.id === id);
}
"""

code_inputs = tokenizer(
    [code],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)

code_inputs = {
    key: tensor.to("cuda")
    for key, tensor in code_inputs.items()
}

with torch.no_grad():
    code_vector = model(
        code_inputs["input_ids"],
        code_inputs["attention_mask"],
    )

similarity = query_vector @ code_vector.T

print("Code vector shape:", code_vector.shape)
print("Similarity:", similarity.item())

candidates = [
    {
        "name": "findUserById",
        "code": """
function findUserById(users, id) {
    return users.find(user => user.id === id);
}
""",
    },
    {
        "name": "addAuthToken",
        "code": """
function addAuthToken(request, token) {
    return {
        ...request,
        headers: {
            ...request.headers,
            Authorization: `Bearer ${token}`,
        },
    };
}
""",
    },
    {
        "name": "limitZoom",
        "code": """
function limitZoom(zoom, minZoom, maxZoom) {
    return Math.min(maxZoom, Math.max(minZoom, zoom));
}
""",
    },
]

code_inputs = tokenizer(
    [candidate["code"] for candidate in candidates],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)

code_inputs = {
    key: tensor.to("cuda")
    for key, tensor in code_inputs.items()
}

with torch.no_grad():
    code_vectors = model(
        code_inputs["input_ids"],
        code_inputs["attention_mask"],
    )

scores = (query_vector @ code_vectors.T)[0]
sorted_indices = scores.argsort(descending=True)

for rank, idx in enumerate(sorted_indices.tolist(), start=1):
    print(
        f"{rank}. {candidates[idx]['name']} "
        f"- {scores[idx].item():.4f}"
    )

query = "look up a person using their identifier"

query_inputs = tokenizer(
    [query],
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
)

query_inputs = {
    key: tensor.to("cuda")
    for key, tensor in query_inputs.items()
}

with torch.no_grad():
    query_vector = model(
        query_inputs["input_ids"],
        query_inputs["attention_mask"],
    )

scores = (query_vector @ code_vectors.T)[0]
sorted_indices = scores.argsort(descending=True)

print("\nQuery:", query)

for rank, idx in enumerate(sorted_indices.tolist(), start=1):
    print(
        f"{rank}. {candidates[idx]['name']} "
        f"- {scores[idx].item():.4f}"
    )

def search(query, top_k=3):
    query_inputs = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    query_inputs = {
        key: tensor.to("cuda")
        for key, tensor in query_inputs.items()
    }

    with torch.no_grad():
        query_vector = model(
            query_inputs["input_ids"],
            query_inputs["attention_mask"],
        )

    scores = (query_vector @ code_vectors.T)[0]

    sorted_indices = scores.argsort(
        descending=True
    )[:top_k]

    results = []

    for idx in sorted_indices.tolist():
        results.append({
            "name": candidates[idx]["name"],
            "score": scores[idx].item(),
        })

    return results

query = "add a bearer token to request headers"
results = search(query, top_k=3)

print("\nQuery:", query)

for rank, result in enumerate(results, start=1):
    print(
        f"{rank}. {result['name']} "
        f"- {result['score']:.4f}"
    )