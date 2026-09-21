# AI Learn 01 — Scaled Dot-Product & Multi-Head Attention from Scratch

First project in the **ai-learn-*** series (AI models & builds phase after classic ML `ml-learn-01`…`12`).

Implement the core Transformer attention equations in plain **NumPy**, train a tiny multi-head model on a synthetic **key→value retrieval** task, and inspect the attention heatmap.

## What you'll learn

- How **scaled dot-product attention** computes `softmax(QKᵀ / √dₖ) V`, and why the `√dₖ` scale matters
- How **multi-head attention** splits projections into parallel heads and recombines with `Wᴼ`
- How a **causal / padding mask** zeros out forbidden positions before softmax
- How attention weights become **interpretable** on a selective-copy / retrieval task (query attends to the matching key)

## Project layout

```
README.md
requirements.txt
attention.py                 # softmax, scaled_dot_product_attention, MultiHeadAttention
run_smoke.py                 # train tiny MHA on synthetic retrieval + write results/
notebooks/attention_scratch.ipynb
results/
  RESULTS.md
  metrics.json
  JSON.shot
  loss_accuracy.png
  attention_heatmap.png
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_smoke.py
```

Smoke run finishes in well under ~2 minutes on CPU, prints train/test accuracy + attention alignment, and refreshes `results/`.

**Notebook walkthrough:**

```bash
jupyter notebook notebooks/attention_scratch.ipynb
```

## Task (smoke demo)

Sequences of length `n_pairs + 1`:

1. Positions `0 … n_pairs-1` each encode a **(key, value)** pair
2. The last position is a **query** asking for one of those keys
3. Target label = the **value** belonging to the queried key

A correct model puts attention mass from the query row onto the matching key column; the classifier then reads out the value.

## Core API

```python
from attention import softmax, scaled_dot_product_attention, MultiHeadAttention

out, weights = scaled_dot_product_attention(Q, K, V, mask=None)
mha = MultiHeadAttention(d_model=32, num_heads=4)
y, attn = mha.forward(X, X, X)   # self-attention
```

## Dependencies

Pinned lightly in `requirements.txt`: **numpy**, **matplotlib**, **jupyter**. PyTorch is optional (notebook comparison cell is skipped if `torch` is absent).
