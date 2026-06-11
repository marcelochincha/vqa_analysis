# Embedding Analysis — Block 2

Exploratory notebook (`check_embeds.ipynb`) that audits the embedding caches used by the pipeline for Block 2, where all answers are integers 1–10.

## Key findings

### Batch size matters for reproducibility

| Model | Batch size | Intra-cluster variance | Notes |
|---|---|---|---|
| AllMPNet | 1 | 0 (bit-identical) | Deterministic: same input → same bits |
| AllMPNet | 32 | ~1e-7 (real) | GPU parallel reduction → different float rounding |
| Qwen3emb4b | 1 | 0 (bit-identical) | Deterministic |

The non-zero `mean_std` observed in batch-1 runs is a **numpy artefact**: `np.std` on bit-identical float32 vectors accumulates floating-point error during mean subtraction. Confirmed via `.view(np.int32)` — integer std is exactly 0.

### Cosine distances between the 10 answer embeddings

| Model | Mean cosine dist | Std |
|---|---|---|
| Random 768-dim (baseline) | 1.00 | 0.03 |
| Qwen3emb4b | 0.13 | 0.08 |
| AllMPNet | 0.33 | 0.10 |

Both models place the 10 numeric answers much closer together than random vectors. The answers are semantically similar short strings — embedding models do not strongly differentiate "1" from "2" in isolation.

### t-SNE instability with identical vectors

When all intra-cluster distances are exactly 0, t-SNE's perplexity search over:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

becomes indeterminate: $\sigma_i$ cannot be resolved by binary search when all neighbours are equidistant. This produces random outlier points despite the clusters being well-defined.

**Fix:** deduplicate to the 10 unique embedding vectors before running t-SNE, then assign all repeated rows to their answer's coordinates.

```python
unique_answers = sorted(df_b2["ANSWER"].dropna().unique())
unique_embs = np.array([cache[get_key(df_b2[df_b2["ANSWER"]==a].iloc[0])]
                        for a in unique_answers])

tsne = TSNE(n_components=2, perplexity=3, random_state=42, init="pca", n_jobs=1)
coords_10 = tsne.fit_transform(unique_embs)
answer_to_coord = {a: coords_10[i] for i, a in enumerate(unique_answers)}
```

With perplexity well below the number of points (3–5 for 10 points), t-SNE is stable and each cluster collapses to a single point.

## Recommendation

For Block 2, cosine similarity over raw embeddings is a weak metric — the 10 answer embeddings are geometrically close regardless of model. Consider numeric distance (`|pred - gt|`) or exact match as the evaluation metric for this block.
