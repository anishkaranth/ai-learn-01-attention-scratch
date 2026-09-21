"""
Scaled dot-product attention and multi-head attention from scratch (NumPy).

Formulas (Vaswani et al., 2017):
  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
  where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V)
    return output, weights


class MultiHeadAttention:
    """Multi-head attention with learned linear projections (NumPy)."""

    def __init__(self, d_model: int, num_heads: int, rng: np.random.Generator | None = None) -> None:
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = rng.normal(0, scale, size=(d_model, d_model))
        self.W_k = rng.normal(0, scale, size=(d_model, d_model))
        self.W_v = rng.normal(0, scale, size=(d_model, d_model))
        self.W_o = rng.normal(0, scale, size=(d_model, d_model))

    def parameters(self) -> dict[str, np.ndarray]:
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        B, H, T, d_k = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(B, T, H * d_k)

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None):
        Qp = np.matmul(Q, self.W_q)
        Kp = np.matmul(K, self.W_k)
        Vp = np.matmul(V, self.W_v)
        Qh = self._split_heads(Qp)
        Kh = self._split_heads(Kp)
        Vh = self._split_heads(Vp)
        context, weights = scaled_dot_product_attention(Qh, Kh, Vh, mask=mask)
        merged = self._merge_heads(context)
        out = np.matmul(merged, self.W_o)
        return out, weights


def causal_mask(seq_len: int) -> np.ndarray:
    """Lower-triangular bool mask (seq, seq) — True = attend."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))
