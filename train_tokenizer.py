from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from data_utils import load_pairs


VOCAB_SIZE = 8000
NUMBER_OF_PAIRS = 5000

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
]


def corpus_iterator(pairs):
    for pair in pairs:
        yield pair["query"]
        yield pair["code"]


print("Loading training pairs...")

train_pairs = load_pairs(
    split="train",
    limit=NUMBER_OF_PAIRS
)

print(f"Loaded {len(train_pairs)} pairs")


tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=ByteLevel.alphabet()
)

print("Training tokenizer...")

tokenizer.train_from_iterator(
    corpus_iterator(train_pairs),
    trainer=trainer,
    length=len(train_pairs) * 2
)




from pathlib import Path

cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[
        ("[CLS]", cls_id),
        ("[SEP]", sep_id),
    ]
)

tokenizer.decoder = ByteLevelDecoder()

Path("artifacts").mkdir(exist_ok=True)

tokenizer.save("artifacts/tokenizer.json")

print("Tokenizer saved")
print("Vocabulary size:", tokenizer.get_vocab_size())

examples = [
    "find a user by id",
    """
function findUserById(users, id) {
    return users.find(user => user.id === id);
}
""",
]

for text in examples:
    encoding = tokenizer.encode(text)

    print("=" * 70)
    print("Original:")
    print(text)

    print("\nTokens:")
    print(encoding.tokens)

    print("\nIDs:")
    print(encoding.ids)

    print("\nDecoded:")
    print(
        tokenizer.decode(
            encoding.ids,
            skip_special_tokens=True
        )
    )