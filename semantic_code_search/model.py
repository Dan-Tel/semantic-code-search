import torch
from torch import nn
import torch.nn.functional as F


class SmallCodeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        cls_token_id,
        sep_token_id,
        embedding_size=256,
        max_sequence_length=256,
        number_of_heads=8,
        number_of_layers=4,
        feedforward_size=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_token_id,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_size,
        )

        self.input_normalization = nn.LayerNorm(
            embedding_size
        )

        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=number_of_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=number_of_layers,
        )

    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum {self.max_sequence_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length,
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        vectors = token_vectors + position_vectors
        vectors = self.input_normalization(vectors)
        vectors = self.input_dropout(vectors)

        padding_mask = attention_mask == 0

        contextual_vectors = self.transformer(
            vectors,
            src_key_padding_mask=padding_mask,
        )

        content_mask = attention_mask.bool()

        content_mask = (
            content_mask
            & (input_ids != self.cls_token_id)
            & (input_ids != self.sep_token_id)
        )

        expanded_mask = content_mask.unsqueeze(-1).float()

        summed_vectors = (
            contextual_vectors * expanded_mask
        ).sum(dim=1)

        token_counts = expanded_mask.sum(
            dim=1
        ).clamp(min=1)

        sequence_vectors = summed_vectors / token_counts

        return F.normalize(
            sequence_vectors,
            p=2,
            dim=1,
        )