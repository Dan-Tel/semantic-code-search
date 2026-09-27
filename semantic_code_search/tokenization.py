from transformers import PreTrainedTokenizerFast


def load_tokenizer(tokenizer_path):
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def tokenize_batch(
    tokenizer,
    texts,
    max_length,
    device
):
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        key: tensor.to(device)
        for key, tensor in inputs.items()
    }