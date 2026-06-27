"""Tokenizers: a SentencePiece language tokenizer (PaliGemma) and the FAST action tokenizer (π0-FAST).

Both are thin wrappers; the SentencePiece model file and the FAST tokenizer parameters are loaded from
``assets/`` and are not committed to the repo (see .gitignore / docs/CHECKPOINTS.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PaligemmaTokenizer:
    """Wraps the PaliGemma SentencePiece model for prompt tokenization."""

    def __init__(self, model_path: str | Path, max_len: int = 48):
        self.max_len = max_len
        self._model_path = Path(model_path)
        self._sp = None  # lazily loaded SentencePieceProcessor

    def _ensure_loaded(self):
        if self._sp is None:
            import sentencepiece as spm

            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"tokenizer model not found at {self._model_path}; see docs/CHECKPOINTS.md"
                )
            self._sp = spm.SentencePieceProcessor(model_file=str(self._model_path))

    def encode(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (token_ids [max_len] int32, mask [max_len] bool), right-padded."""
        self._ensure_loaded()
        ids = self._sp.encode(prompt, out_type=int)[: self.max_len]
        mask = np.zeros(self.max_len, dtype=bool)
        out = np.zeros(self.max_len, dtype=np.int32)
        out[: len(ids)] = ids
        mask[: len(ids)] = True
        return out, mask


class FastActionTokenizer:
    """FAST (Frequency-space Action Sequence Tokenization) for the autoregressive π0-FAST model.

    FAST applies a DCT to action chunks and entropy-codes the result into a compact discrete vocabulary, letting
    a standard LLM decode actions autoregressively. This stub defines the interface; the codebook is loaded from
    ``assets/`` at runtime.
    """

    def __init__(self, vocab_size: int = 1024, horizon: int = 50, action_dim: int = 7):
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.action_dim = action_dim

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """actions: [horizon, action_dim] -> token ids [num_tokens] int32."""
        raise NotImplementedError("FAST encode: implement DCT + entropy coding. See docs/ROADMAP.md")

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """token ids -> actions [horizon, action_dim] float32."""
        raise NotImplementedError("FAST decode: implement inverse entropy coding + IDCT. See docs/ROADMAP.md")
