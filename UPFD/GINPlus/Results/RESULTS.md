# GIN⁺ on UPFD Gossipcop: results and interpretation

Source run: Kaggle `gourasri/gin-plus-upfd` (T4, 75 min). Split: the official UPFD split (1,092 train, 546 validation, 3,826 test); one test graph is worth about 0.03 accuracy points.

Final numbers are the mean ± std over 10 seeds. Ablations use 5 seeds and sweeps use 3. Comparisons with the other models use `UPFD/GCNPlus` and `UPFD/GraphGPS`.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC | Params |
|---|---|---|---|---|
| Majority class | 0.501 | – | – | – |
| Cascade size only | 0.752 | – | – | – |
| News-node features only | 0.937 | – | – | – |
| Plain GIN (2 layers, no tricks) | 0.930 ± 0.066 | 0.930 ± 0.066 | 0.966 ± 0.039 | – |
| **GIN⁺ (tuned)** | **0.972 ± 0.003** | **0.972 ± 0.003** | **0.993 ± 0.001** | **12.8k** |
| GCN⁺ (tuned) | 0.970 ± 0.002 | 0.970 ± 0.002 | 0.990 ± 0.003 | 400k |
| GraphGPS (tuned) | 0.971 ± 0.005 | 0.971 ± 0.005 | 0.985 ± 0.004 | 188k |

The best configuration had hidden width 32, **1 layer**, dropout 0.1, max readout, learning rate 1e-2, no weight decay, and all tricks on. Its best mean validation accuracy was 0.972.

- **All three models tie on UPFD** (0.970–0.972, within one std). GIN⁺ has the best AUC (0.993) and does it with **31× fewer parameters than GCN⁺** and a single message-passing layer.
- The search picking one layer agrees with the input ablation below: on Gossipcop, message passing adds little.
- The plain GIN is much better than the plain GCN (0.930 vs 0.615), but it is unstable (std 0.066).

## 2. Which tricks matter? (`ginp_ablation.png`)

| Variant (5 seeds) | Accuracy | Δ vs full |
|---|---|---|
| Full GIN⁺ | 0.970 | – |
| No normalization | 0.976 | +0.5 |
| No dropout | 0.973 | +0.3 |
| No residual | 0.973 | +0.2 |
| No FFN | 0.972 | +0.1 |
| No positional encoding | 0.975 | +0.4 |
| All tricks off | 0.965 | −0.5 |

**No trick matters for a 1-layer GIN.** Every single removal is within noise, and even all tricks off costs only 0.5 points. The tricks matter for stability in deeper models (section 5) and for GCN (−19 points with all off).

## 3. What does the model use?

| Input (retrained) | Accuracy |
|---|---|
| Full graph | 0.970 |
| No edges | **0.979** |
| Constant features | 0.815 |
| News node only | 0.954 |

| Feature block masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| News: text (300) | 0.001 | 0.972 |
| News: profile (10) | 0.000 | 0.973 |
| **Users: text (300)** | **0.263** | **0.803** |
| Users: profile (10) | 0.070 | 0.938 |

As with GCN⁺, removing the edges helps slightly and the users' text embeddings carry the decision. GIN⁺ also uses the user profiles more than GCN⁺ does (0.938 vs 0.963 when masked). Masking user text hurts it less (0.803 vs 0.501).

## 4. Readout comparison (`ginp_readout.png`)

| Readout | Accuracy | AUC | Peak memory | s / epoch |
|---|---|---|---|---|
| **max (selected)** | **0.970** | 0.993 | 354 MB | 0.16 |
| SAGPool | 0.968 | 0.990 | 354 MB | 0.23 |
| news-aware | 0.968 | 0.990 | 354 MB | 0.21 |
| mean | 0.958 | 0.986 | 353 MB | 0.19 |
| DiffPool | 0.957 | 0.982 | 476 MB | 0.26 |
| MinCutPool | 0.957 | 0.984 | 441 MB | 0.28 |
| add | 0.953 | 0.981 | 353 MB | 0.16 |

- Max pooling wins, as it does for GCN⁺ and GraphGPS. Mean and sum lose 1.2–1.7 points here, more than for GCN⁺.
- The dense pools use 25–35% more memory and are among the worst.

## 5. Depth and dropout (`ginp_depth.png`, `ginp_dropout.png`)

| Layers | With residual | Without residual |
|---|---|---|
| 2 | 0.968 | 0.961 |
| 4 | 0.973 | 0.968 |
| 8 | 0.974 | **0.782 ± 0.035** |
| 16 | 0.973 | **0.791 ± 0.017** |

- **This is the clearest residual effect in the project.** Without residuals, GIN collapses at 8+ layers (−19 points). With them, 16 layers is as good as 4.
- Dropout from 0 to 0.3 gives 0.967–0.969; 0.5 is worst at 0.960.

## 6. Embeddings (`ginp_tsne.png`)

The silhouette score for fake vs real rises from 0.149 (raw node features) to 0.492 (GIN⁺ embedding). GCN⁺ scores 0.515 and GraphGPS 0.514.

## 7. Robustness, with no retraining (`ginp_robustness.png`)

| Damage level | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| Edges removed | 0.973 | 0.973 | 0.973 | 0.972 | 0.969 |
| Feature noise σ | 0.973 | 0.936 | **0.651** | 0.577 | 0.645 |

- Edge loss does essentially nothing: −0.4 points at 50% removed, against −3.6 for GCN⁺. A 1-layer model barely touches the edges.
- **The most fragile model to feature noise.** σ = 0.2 already drops it to 0.65, against 0.94 for GCN⁺ and 0.80 for GraphGPS. Max pooling picks the most extreme user, and noise creates extreme values.

## 8. Errors by topology (`ginp_topology.png`)

| Cascade size | Accuracy | | Star-ness | Accuracy |
|---|---|---|---|---|
| 3–19 nodes | **0.964** | | 0.03–0.67 (chains) | **0.953** |
| 20–49 | 0.973 | | 0.67–0.85 | 0.977 |
| 50–83 | 0.981 | | 0.85–0.95 | 0.972 |
| 84–199 | 0.973 | | 0.95–0.99 (stars) | 0.990 |

Depth ≤ 3 hops scores 0.975 and > 3 hops 0.958. GIN⁺ is weakest on deep, chain-like cascades, which a 1-layer model cannot see beyond its first hop. The small-cascade and chain patterns match the other two models.

## 9. Explainability: GNNExplainer (`ginp_explained_cascades.png`)

| Users removed | Mean drop in confidence |
|---|---|
| Top 10% by GNNExplainer | **0.0235** |
| Random 10% | 0.0032 |

- **GIN⁺ is the only UPFD model whose explanations pass the deletion test**, at 7.4× the random drop. GCN⁺ gets 0.1×, and GraphGPS has a random drop of about 0.
- With one layer and max pooling, a few users really do decide the output, so a node mask can find them.
- The news node ranks below 21 other nodes on average.

## 10. Takeaways

1. GIN⁺ ties the other models at **0.972 ± 0.003** (AUC 0.993) with **12.8k parameters and 1 layer**, the most efficient UPFD model.
2. Its tricks do not matter at 1 layer. Residuals become critical when it goes deep (0.79 without them at 8+ layers).
3. Max readout wins, edges are unnecessary, and the users' text is the main signal.
4. It is robust to edge loss but the most fragile to feature noise.
5. It is the only UPFD model with faithful GNNExplainer explanations.
