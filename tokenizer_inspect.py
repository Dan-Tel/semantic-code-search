from transformers import AutoTokenizer


MODEL_NAME = "microsoft/codebert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

query = "find a user by id"

encoded = tokenizer(
    query,
    max_length=16,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
)

print("Original query:")
print(query)

print("\nTokens:")
print(tokens)

print("\nInput IDs:")
print(input_ids)

print("\nAttention mask:")
print(attention_mask)

print("\nShapes:")
print("input_ids:", input_ids.shape)
print("attention_mask:", attention_mask.shape)

code = """
function findUserById(
    users: User[],
    id: number
): User | undefined {
    return users.find(user => user.id === id);
}
"""

encoded_code = tokenizer(
    code,
    max_length=32,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

code_tokens = tokenizer.convert_ids_to_tokens(
    encoded_code["input_ids"][0]
)

print("\nCode tokens:")
print(code_tokens)

print("\nCode input IDs:")
print(encoded_code["input_ids"])

print("\nCode attention mask:")
print(encoded_code["attention_mask"])

print("\nDecoded:")
print(
    tokenizer.decode(
        encoded_code["input_ids"][0],
        skip_special_tokens=True
    )
)