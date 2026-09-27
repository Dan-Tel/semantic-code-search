from transformers import PreTrainedTokenizerFast


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="artifacts/tokenizer.json",
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)

queries = [
    "find user",
    "remove authentication token from request headers",
]

encoded = tokenizer(
    queries,
    padding="max_length",
    truncation=True,
    max_length=12,
    return_tensors="pt",
)

# print("Input IDs:")
# print(encoded["input_ids"])

# print("\nAttention mask:")
# print(encoded["attention_mask"])

# print("\nShape:")
# print(encoded["input_ids"].shape)

# print("\nTokens:")

# for input_ids in encoded["input_ids"]:
#     tokens = tokenizer.convert_ids_to_tokens(
#         input_ids.tolist()
#     )

#     print(tokens)




import torch
from torch import nn


torch.manual_seed(42)

EMBEDDING_SIZE = 256

token_embedding = nn.Embedding(
    num_embeddings=len(tokenizer),
    embedding_dim=EMBEDDING_SIZE,
    padding_idx=tokenizer.pad_token_id,
)

token_vectors = token_embedding(
    encoded["input_ids"]
)

# print("\nVocabulary size:")
# print(len(tokenizer))

# print("\nInput shape:")
# print(encoded["input_ids"].shape)

# print("\nToken vectors shape:")
# print(token_vectors.shape)

# print("\nFirst token vector:")
# print(token_vectors[0, 0, :10])

# print("\nPadding vector:")
# print(token_vectors[0, 4, :10])

MAX_SEQUENCE_LENGTH = 256

position_embedding = nn.Embedding(
    num_embeddings=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBEDDING_SIZE,
)

batch_size, sequence_length = encoded["input_ids"].shape

positions = torch.arange(
    sequence_length,
    device=encoded["input_ids"].device,
)

positions = positions.unsqueeze(0).expand(
    batch_size,
    sequence_length,
)

position_vectors = position_embedding(positions)

combined_vectors = token_vectors + position_vectors

# print("\nPositions:")
# print(positions)

# print("\nToken vectors shape:")
# print(token_vectors.shape)

# print("\nPosition vectors shape:")
# print(position_vectors.shape)

# print("\nCombined vectors shape:")
# print(combined_vectors.shape)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=EMBEDDING_SIZE,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.1,
    activation="gelu",
    batch_first=True,
)

transformer_encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=4,
)

padding_mask = encoded["attention_mask"] == 0

# print("\nPadding mask:")
# print(padding_mask)



contextual_vectors = transformer_encoder(
    combined_vectors,
    src_key_padding_mask=padding_mask,
)

# print("\nBefore Transformer:")
# print(combined_vectors.shape)

# print("\nAfter Transformer:")
# print(contextual_vectors.shape)




import torch.nn.functional as F

content_mask = encoded["attention_mask"].bool()

content_mask = (
    content_mask
    & (encoded["input_ids"] != tokenizer.cls_token_id)
    & (encoded["input_ids"] != tokenizer.sep_token_id)
)

expanded_mask = content_mask.unsqueeze(-1).float()

masked_vectors = contextual_vectors * expanded_mask

summed_vectors = masked_vectors.sum(dim=1)

token_counts = expanded_mask.sum(dim=1).clamp(min=1)

sequence_vectors = summed_vectors / token_counts

sequence_vectors = F.normalize(
    sequence_vectors,
    p=2,
    dim=1,
)

print("\nSequence vectors shape:")
print(sequence_vectors.shape)

print("\nVector lengths:")
print(torch.linalg.vector_norm(sequence_vectors, dim=1))

similarity_matrix = (
    sequence_vectors
    @ sequence_vectors.T
)

print("\nSimilarity matrix:")
print(similarity_matrix)

from small_code_encoder import SmallCodeEncoder

model = SmallCodeEncoder(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    cls_token_id=tokenizer.cls_token_id,
    sep_token_id=tokenizer.sep_token_id,
)

model.eval()

with torch.no_grad():
    sequence_vectors = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nOutput shape:")
print(sequence_vectors.shape)

print("\nVector lengths:")
print(torch.linalg.vector_norm(sequence_vectors, dim=1))

print("\nNumber of parameters:")
print(f"{number_of_parameters:,}")