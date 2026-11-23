# GraphGPS on UPFD Gossipcop: results and interpretation

Source run: Kaggle `gouravanirudh1105/graphgps-upfd` (T4, 170 min). Split: the official UPFD split (1,092 train, 546 validation, 3,826 test); one test graph is worth about 0.03 accuracy points.

Final numbers are the mean ± std over 10 seeds. Ablations use 5 seeds and sweeps use 3. Comparisons with the other models use `UPFD/GCNPlus` and `UPFD/GINPlus`.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC | Params |
|---|---|---|---|---|
| Majority class | 0.501 | – | – | – |
| Cascade size only | 0.752 | – | – | – |
| News-node features only | 0.937 | – | – | – |
| Plain GPS (2 layers, no tricks) | 0.827 ± 0.085 | 0.823 ± 0.088 | 0.907 ± 0.058 | – |
| **GraphGPS (tuned)** | **0.971 ± 0.005** | **0.971 ± 0.005** | **0.985 ± 0.004** | **188k** |
| GCN⁺ (tuned) | 0.970 ± 0.002 | 0.970 ± 0.002 | 0.990 ± 0.003 | 400k |
| GIN⁺ (tuned) | 0.972 ± 0.003 | 0.972 ± 0.003 | 0.993 ± 0.001 | 12.8k |

The best configuration had hidden width 64, 4 layers, 4 heads, dropout 0, sum readout, learning rate 1e-2, weight decay 1e-5, and all tricks on. Its best mean validation accuracy was 0.976.

- **GraphGPS ties GCN⁺ and GIN⁺** (0.970–0.972). It has the lowest AUC of the three and the largest seed variance. Global attention buys nothing on UPFD, unlike on PROTEINS (+1.7 over GIN⁺) and FRANKENSTEIN (+5.4).
- Tuning matters: the plain GPS is unstable (0.827 ± 0.085).

## 2. Which components matter? (`gps_ablation.png`)

| Variant (5 seeds) | Accuracy | Δ vs full |
|---|---|---|
| Full GraphGPS | 0.969 | – |
| **No normalization** | **0.551 ± 0.112** | **−41.9** |
| No residual | 0.964 | −0.5 |
| No FFN | 0.979 | +1.0 |
| No positional encoding | 0.972 | +0.3 |
| No attention (= GIN⁺ block) | 0.962 | −0.8 |
| No local MPNN (pure Transformer) | 0.970 | +0.1 |
| All tricks off | 0.763 ± 0.057 | −20.6 |

There is no "no dropout" row because the tuned dropout is already 0.

- **Normalization is essential for GraphGPS**: without BatchNorm, training collapses to chance. Attention over cascades with up to 199 nodes and a hub at the root produces badly scaled activations that only normalization keeps in check.
- Attention vs local MPNN: neither branch matters much here (−0.8 / +0.1), the opposite of PROTEINS, where each was worth 3–4 points.
- The FFN hurts slightly, and positional encoding adds nothing, as for GCN⁺.

## 3. What does the model use?

| Input (retrained) | Accuracy |
|---|---|
| Full graph | 0.969 |
| No edges | **0.975** |
| Constant features | 0.799 |
| News node only | 0.958 |

| Feature block masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| News: text (300) | 0.005 | 0.970 |
| News: profile (10) | 0.000 | 0.973 |
| **Users: text (300)** | **0.498** | **0.501** |
| Users: profile (10) | 0.126 | 0.922 |

This is the same picture as the other two models: edges are unnecessary, and masking the users' text drops the model to chance. GraphGPS relies on user profiles the most of the three (0.922 when masked).

## 4. Readout comparison (`gps_readout.png`)

| Readout | Accuracy | AUC |
|---|---|---|
| **max** | **0.974** | 0.989 |
| mean | 0.973 | 0.990 |
| news-aware | 0.972 | 0.988 |
| DiffPool | 0.970 | 0.985 |
| add (selected) | 0.969 | 0.984 |
| SAGPool | 0.967 | 0.982 |
| MinCutPool | 0.959 ± 0.027 | 0.982 |

- Max is best on test for all three UPFD models. Here the search picked sum on validation, which is 0.5 points behind max.
- Peak memory is 727 MB for every readout, because the dense attention dominates.

## 5. Depth and dropout (`gps_depth.png`, `gps_dropout.png`)

| Layers | With residual | Without residual |
|---|---|---|
| 2 | **0.975** | 0.976 |
| 4 | 0.973 | 0.958 |
| 8 | 0.956 | 0.871 ± 0.048 |
| 16 | **0.804** | 0.784 |

- **Residuals do not save deep GraphGPS**: 16 layers falls to 0.80 even with them. GCN⁺ and GIN⁺ stay at 0.97. Stacked global attention over-smooths the whole cascade, since every layer mixes every node.
- Dropout: 0.1 and 0.2 are unstable (0.911 ± 0.094 and 0.913 ± 0.090, with some seeds collapsing). The values 0, 0.3 and 0.5 are stable at 0.963–0.970.

## 6. Embeddings (`gps_tsne.png`)

The silhouette score rises from 0.149 (raw node features) to 0.514, the same as GCN⁺ (0.515) and GIN⁺ (0.492).

## 7. Robustness, with no retraining (`gps_robustness.png`)

| Damage level | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| Edges removed | 0.972 | 0.973 | 0.973 | 0.973 | 0.973 | 0.963 |
| Feature noise σ | 0.972 | 0.956 | 0.801 | 0.643 | – | 0.560 |

- Edge loss costs nothing up to 40% removed, and −0.9 points at 50%. Attention does not depend on the edges.
- Feature noise is harmful: 0.80 at σ = 0.2. That is more robust than GIN⁺ (0.65) but much less than GCN⁺ (0.94), whose degree-normalised averaging smooths the noise.

## 8. Errors by topology (`gps_topology.png`)

| Cascade size | Accuracy | | Star-ness | Accuracy |
|---|---|---|---|---|
| 3–19 nodes | **0.963** | | 0.03–0.67 (chains) | **0.949** |
| 20–49 | 0.973 | | 0.67–0.85 | 0.977 |
| 50–83 | 0.975 | | 0.85–0.95 | 0.980 |
| 84–199 | 0.978 | | 0.95–0.99 (stars) | 0.983 |

Depth ≤ 3 hops scores 0.973 and > 3 hops 0.967. The errors fall in the same places as for the other models: small and chain-like cascades.

## 9. Explainability: GNNExplainer (`gps_explained_cascades.png`)

| Users removed | Mean drop in confidence |
|---|---|
| Top 10% by GNNExplainer | 0.0103 |
| Random 10% | −0.0003 |

The top-ranked users do matter more than random ones, whose removal changes nothing. But the effect is small: a 1-point drop in confidence. The news node ranks below 33 other nodes on average. Feature masking (section 3) remains the clearer explanation.

## 10. Efficiency

| Model (selected readout) | Params | s / epoch | Inference (test set) | Peak memory | Notebook |
|---|---|---|---|---|---|
| GCN⁺ (SAGPool) | 400k | 0.34 | 0.17 s | 482 MB | 84 min |
| GIN⁺ (max) | 12.8k | 0.16 | 0.02 s | 354 MB | 75 min |
| GraphGPS (sum) | 188k | 0.58 | 0.47 s | 727 MB | 170 min |

GraphGPS is the most expensive model on every axis: 3.6× GIN⁺'s epoch time and 2× its memory, for the same accuracy.

## 11. Takeaways

1. GraphGPS reaches **0.971 ± 0.005**, a tie with GCN⁺ and GIN⁺. On UPFD, global attention adds cost but no accuracy.
2. **Normalization is indispensable**: −42 points without it.
3. Deep GraphGPS degrades even with residuals (0.80 at 16 layers).
4. It is robust to edge loss and moderately fragile to feature noise.
5. Across the three datasets, GraphGPS wins where structure carries the signal (PROTEINS, FRANKENSTEIN) and only ties where it doesn't (UPFD).
