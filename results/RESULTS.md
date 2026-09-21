# Smoke results — ai-learn-01-attention-scratch

**Task:** synthetic key→value retrieval (selective copy) with multi-head self-attention.

**Seed:** `42`

## Headline metrics

| Metric | Value |
|--------|------:|
| Final train accuracy | 1.0000 |
| Final test accuracy | 1.0000 |
| Attention alignment (query→match pos) | 0.9922 |
| Final train loss | 0.0129 |
| Runtime (s) | 1.80 |
| Unit checks | PASS |

## Config

- `d_model=48`, `num_heads=4`, `seq_len=5`, `vocab=8`
- train `768` / test `128`, batch `64`, epochs `80`, lr `0.08`

## Plots

- [`loss_accuracy.png`](loss_accuracy.png)
- [`attention_heatmap.png`](attention_heatmap.png)

## Takeaway

The query token learns to put mass on the matching key position; the classifier reads out the associated value.
