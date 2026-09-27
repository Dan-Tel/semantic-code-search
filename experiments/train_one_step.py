import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizerFast

from semantic_code_search.data import load_pairs
from semantic_code_search.model import SmallCodeEncoder


torch.manual_seed(42)

BATCH_SIZE = 4
QUERY_LENGTH = 64
CODE_LENGTH = 256
TEMPERATURE = 0.1

pairs = load_pairs(
    split="train",
    limit=BATCH_SIZE,
)

queries = [
    pair["query"]
    for pair in pairs
]

codes = [
    pair["code"]
    for pair in pairs
]

for index, pair in enumerate(pairs):
    print(
        f"{index}: {pair['query']} "
        f"↔ {pair['name']}"
    )

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="artifacts/tokenizer.json",
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)

query_batch = tokenizer(
    queries,
    padding="max_length",
    truncation=True,
    max_length=QUERY_LENGTH,
    return_tensors="pt",
)

code_batch = tokenizer(
    codes,
    padding="max_length",
    truncation=True,
    max_length=CODE_LENGTH,
    return_tensors="pt",
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



model = SmallCodeEncoder(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    cls_token_id=tokenizer.cls_token_id,
    sep_token_id=tokenizer.sep_token_id,
)

model = model.to(device)
model.train()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)


print("Device:", device)
print("Model device:", next(model.parameters()).device)


query_batch = {
    key: value.to(device)
    for key, value in query_batch.items()
}

code_batch = {
    key: value.to(device)
    for key, value in code_batch.items()
}



query_vectors = model(
    input_ids=query_batch["input_ids"],
    attention_mask=query_batch["attention_mask"],
)

code_vectors = model(
    input_ids=code_batch["input_ids"],
    attention_mask=code_batch["attention_mask"],
)


# similarity_matrix = query_vectors @ code_vectors.T

# logits = similarity_matrix / TEMPERATURE

# labels = torch.arange(
#     BATCH_SIZE,
#     device=device,
# )

# loss = F.cross_entropy(
#     logits,
#     labels,
# )

# print("\nSimilarity matrix:")
# print(similarity_matrix.detach().cpu())

# print("\nLabels:")
# print(labels)

# print("\nLoss:")
# print(loss.item())


# parameter = next(model.parameters())
# parameter_before = parameter.detach().clone()

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# parameter_after = parameter.detach()

# max_change = (
#     parameter_after - parameter_before
# ).abs().max().item()

# print("Maximum parameter change:", max_change)

query_input_ids = query_batch["input_ids"]
query_attention_mask = query_batch["attention_mask"]

code_input_ids = code_batch["input_ids"]
code_attention_mask = code_batch["attention_mask"]

labels = torch.arange(
    BATCH_SIZE,
    device=device,
)

for step in range(20):
    query_vectors = model(
        query_input_ids,
        query_attention_mask
    )

    code_vectors = model(
        code_input_ids,
        code_attention_mask
    )

    similarity_matrix = query_vectors @ code_vectors.T
    logits = similarity_matrix / TEMPERATURE
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step + 1}: loss = {loss.item():.4f}")




model.eval()

with torch.no_grad():
    query_vectors = model(
        query_input_ids,
        query_attention_mask
    )

    code_vectors = model(
        code_input_ids,
        code_attention_mask
    )

    similarity_matrix = query_vectors @ code_vectors.T
    predicted_indices = similarity_matrix.argmax(dim=1)

print("Similarity matrix:")
print(similarity_matrix.cpu())

print("Expected:", labels.cpu())
print("Predicted:", predicted_indices.cpu())



# embedding_gradient = (
#     model.token_embedding.weight.grad
# )

# print("\nGradient exists:")
# print(embedding_gradient is not None)

# print("\nGradient norm:")
# print(embedding_gradient.norm().item())

# print("\nPAD gradient norm:")
# print(
#     embedding_gradient[
#         tokenizer.pad_token_id
#     ].norm().item()
# )

# print(model.token_embedding)
# print(embedding_gradient)