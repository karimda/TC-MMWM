"""
Language Encoder for TC-MMWM

Encodes natural language instructions into a structured causal latent embedding.
Language is treated as a constraint-defining modality that directly shapes
latent state transitions rather than an auxiliary feature.

Corresponds to:
- Section 2.2 (Multimodal Causal State Space)
- Section 3.5 (Language Constraint Compliance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageEncoder(nn.Module):
    """
    Language encoder based on pretrained transformer embeddings
    followed by a causal projection head.

    Input:
        - Token embeddings of shape (B, T, D_l)

    Output:
        - Language latent embedding z_l of shape (B, D_latent)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        latent_dim: int = 128,
        max_tokens: int = 64,
        pooling: str = "mean"
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.max_tokens = max_tokens
        self.pooling = pooling

        # Projection network to causal language latent
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        self._initialize_weights()

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for language encoder.

        Args:
            token_embeddings:
                Tensor of shape (B, T, embedding_dim),
                obtained from a frozen pretrained language model.

        Returns:
            z_l: Language latent representation (B, latent_dim)
        """
        if self.pooling == "mean":
            pooled = token_embeddings.mean(dim=1)
        elif self.pooling == "max":
            pooled, _ = token_embeddings.max(dim=1)
        elif self.pooling == "cls":
            pooled = token_embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        z_l = self.fc(pooled)
        return z_l

    def _initialize_weights(self):
        """
        Xavier initialization ensures stable optimization
        and prevents dominance of language modality.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
