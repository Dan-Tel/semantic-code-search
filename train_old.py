from pathlib import Path

from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F

from data_utils import load_pairs
from small_code_encoder import SmallCodeEncoder


BATCH_SIZE = 4
QUERY_LENGTH = 64
CODE_LENGTH = 256
TEMPERATURE = 0.1

TRAIN_LIMIT = 5000
VALIDATION_LIMIT = 500
BATCH_SIZE = 32
NUM_EPOCHS = 10

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



train_pairs = load_pairs(
    split="train",
    limit=TRAIN_LIMIT
)

train_loader = DataLoader(
    train_pairs,
    batch_size=BATCH_SIZE,
    shuffle=True
)


validation_pairs = load_pairs(
    split="validation",
    limit=VALIDATION_LIMIT
)

validation_loader = DataLoader(
    validation_pairs,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training pairs:", len(train_pairs))
print("Validation pairs:", len(validation_pairs))
print("Validation batches:", len(validation_loader))



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
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)




best_mrr = float("-inf")

best_checkpoint_path = Path(
    "checkpoints/small_code_encoder_q2c_best.pt"
)


for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        queries = list(batch["query"])
        codes = list(batch["code"])

        query_inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        code_inputs = tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        query_inputs = {
            key: tensor.to(device)
            for key, tensor in query_inputs.items()
        }

        code_inputs = {
            key: tensor.to(device)
            for key, tensor in code_inputs.items()
        }

        query_vectors = model(
            query_inputs["input_ids"],
            query_inputs["attention_mask"]
        )

        code_vectors = model(
            code_inputs["input_ids"],
            code_inputs["attention_mask"]
        )

        similarity_matrix = query_vectors @ code_vectors.T
        logits = similarity_matrix / TEMPERATURE

        batch_size = logits.size(0)

        labels = torch.arange(
            batch_size,
            device=device
        )

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # print(
    #     f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
    #     f"average loss = {average_loss:.4f}"
    # )

    model.eval()
    total_validation_loss = 0.0

    correct = 0
    total = 0

    all_query_vectors = []
    all_code_vectors = []

    with torch.no_grad():
        for batch in validation_loader:
            queries = list(batch["query"])
            codes = list(batch["code"])

            query_inputs = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )

            code_inputs = tokenizer(
                codes,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            query_inputs = {
                key: tensor.to(device)
                for key, tensor in query_inputs.items()
            }

            code_inputs = {
                key: tensor.to(device)
                for key, tensor in code_inputs.items()
            }

            query_vectors = model(
                query_inputs["input_ids"],
                query_inputs["attention_mask"]
            )

            code_vectors = model(
                code_inputs["input_ids"],
                code_inputs["attention_mask"]
            )

            all_query_vectors.append(
                query_vectors.cpu()
            )

            all_code_vectors.append(
                code_vectors.cpu()
            )

            similarity_matrix = query_vectors @ code_vectors.T
            logits = similarity_matrix / TEMPERATURE

            labels = torch.arange(
                logits.size(0),
                device=device
            )

            predicted = logits.argmax(dim=1)

            correct += (
                predicted == labels
            ).sum().item()

            total += labels.size(0)

            loss = F.cross_entropy(logits, labels)
            total_validation_loss += loss.item()


    all_query_vectors = torch.cat(
        all_query_vectors,
        dim=0
    )

    all_code_vectors = torch.cat(
        all_code_vectors,
        dim=0
    )

    full_similarity_matrix = (
        all_query_vectors @ all_code_vectors.T
    )

    # print(
    #     "Full similarity matrix:",
    #     full_similarity_matrix.shape
    # )



    predicted_indices = (
        full_similarity_matrix.argmax(dim=1)
    )

    expected_indices = torch.arange(
        len(validation_pairs)
    )

    correct = (
        predicted_indices == expected_indices
    ).sum().item()

    full_top1_accuracy = (
        correct / len(validation_pairs)
    )

    # print("Correct:", correct)
    # print(
    #     f"Full Top-1 accuracy: "
    #     f"{full_top1_accuracy:.2%}"
    # )


    top5_indices = full_similarity_matrix.topk(
        k=5,
        dim=1
    ).indices

    # print("Top-5 shape:", top5_indices.shape)

    expected_column = expected_indices.unsqueeze(1)

    hits_at_5 = (
        top5_indices == expected_column
    ).any(dim=1)

    recall_at_5 = hits_at_5.float().mean().item()

    # print(
    #     f"Recall@5: "
    #     f"{recall_at_5:.2%}"
    # )


    sorted_indices = full_similarity_matrix.argsort(
        dim=1,
        descending=True
    )

    matches = (
        sorted_indices
        == expected_indices.unsqueeze(1)
    )

    ranks = (
        matches.float().argmax(dim=1) + 1
    )

    # print("First 10 ranks:", ranks[:10])

    reciprocal_ranks = 1.0 / ranks.float()
    mrr = reciprocal_ranks.mean().item()

    # print(f"MRR: {mrr:.4f}")


    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"train loss: {average_loss:.4f} | "
        f"validation MRR: {mrr:.4f}"
    )


    # 3. Сохранение лучшей модели
    if mrr > best_mrr:
        best_mrr = mrr

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mrr": mrr,
            },
            best_checkpoint_path
        )

        print("New best model saved")




# average_validation_loss = (
#     total_validation_loss / len(validation_loader)
# )

# print(
#     f"Validation loss: "
#     f"{average_validation_loss:.4f}"
# )

# validation_accuracy = correct / total

# print(
#     f"Validation accuracy: "
#     f"{validation_accuracy:.2%}"
# )




# checkpoint_path = Path(
#     "checkpoints/small_code_encoder.pt"
# )

# checkpoint_path.parent.mkdir(
#     parents=True,
#     exist_ok=True
# )

# torch.save(
#     {
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "epochs": NUM_EPOCHS,
#         "train_limit": TRAIN_LIMIT,
#         "validation_limit": VALIDATION_LIMIT,
#     },
#     checkpoint_path
# )

# print("Saved:", checkpoint_path)