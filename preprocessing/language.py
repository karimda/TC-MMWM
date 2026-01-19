"""
Language preprocessing and encoding module for TC-MMWM.

This file implements:
- Text preprocessing and tokenization
- Transformer-based language encoding
- Projection to language-induced causal contribution Δz_t^(lang)

Aligned with:
Section 2.2.2.2 Language-Induced State Constraints
Equation (3): Δz_t^(lang) = g_l(l_t, z_{t-1})
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    raise ImportError(
        "HuggingFace transformers is required for language preprocessing. "
        "Install via: pip install transformers"
    )


# ------------------------------------------------------------
# Language Preprocessing
# ------------------------------------------------------------

class LanguagePreprocessor:
    """
    Preprocess language instructions using a pretrained tokenizer.

    Input:
        - Raw instruction string or list of strings

    Output:
        - Tokenized representation:
            input_ids: (B, max_seq_len)
            attention_mask: (B, max_seq_len)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_seq_len: int = 32,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of language instructions.

        Args:
            texts: list of instruction strings

        Returns:
            tokenized dict with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return encoded


# ------------------------------------------------------------
# Language Encoder
# ------------------------------------------------------------

class LanguageEncoder(nn.Module):
    """
    Transformer-based language encoder producing Δz_t^(lang).

    Language is treated as a *structural constraint* on state transitions,
    not merely as an auxiliary observation.

    Implements g_l(l_t, z_{t-1}) from Eq. (3).

    Output:
        - Δz_t^(lang) ∈ R^{latent_dim}
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        latent_dim: int = 256,
        freeze_language_model: bool = True,
    ):
        super().__init__()

        self.language_model = AutoModel.from_pretrained(model_name)
        hidden_dim = self.language_model.config.hidden_size

        if freeze_language_model:
            for p in self.language_model.parameters():
                p.requires_grad = False

        # Projection into latent causal space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        # Conditioning on previous latent state z_{t-1}
        self.state_condition = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        tokenized_inputs: Dict[str, torch.Tensor],
        prev_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokenized_inputs:
                - input_ids: (B, L)
                - attention_mask: (B, L)
            prev_latent: z_{t-1} (B, latent_dim) or None

        Returns:
            Δz_t^(lang): (B, latent_dim)
        """

        outputs = self.language_model(
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
        )

        # Use [CLS] token representation as sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, hidden_dim)

        delta_z = self.projection(cls_embedding)

        # Language biases state evolution through causal conditioning
        if prev_latent is not None:
            delta_z = delta_z + self.state_condition(prev_latent)

        return delta_z


# ------------------------------------------------------------
# Utility: Instruction Statistics
# ------------------------------------------------------------

def compute_instruction_statistics(tokenized_batch: Dict[str, torch.Tensor]) -> dict:
    """
    Compute basic statistics over tokenized instructions.

    Useful for:
        - Dataset cards
        - Artifact evaluation checklist
        - Debugging instruction length distributions
    """
    attention_mask = tokenized_batch["attention_mask"]
    lengths = attention_mask.sum(dim=1).cpu().tolist()

    return {
        "mean_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
    }
