# GraphGPS on PROTEINS: results and interpretation

Source run: Kaggle `gouravanirudh1105/graphgps-proteins` (T4, 101 min). Split: stratified 80/10/10 with `random_state=0` (890 / 111 / 112 graphs). Labels: y=0 is enzyme (59.8%) and y=1 is non-enzyme.

The test set has 112 proteins, so **one graph is worth 0.9 accuracy points**. Differences under about 2 points are within seed noise. Final numbers are the mean ± std over 10 seeds; ablations use 5 seeds and sweeps 3.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.598 | – | – |
| Protein size only | 0.741 | – | – |
| Bag of nodes | 0.759 | – | – |
| Plain GPS (2 layers, no tricks) | 0.770 ± 0.022 | 0.750 ± 0.030 | 0.841 ± 0.027 |
| **GraphGPS (tuned)** | **0.778 ± 0.016** | **0.765 ± 0.015** | **0.838 ± 0.032** |

The best configuration had hidden width 32, 3 layers, 2 attention heads, dropout 0.2, mean readout, learning rate 1e-2, weight decay 1e-5, and all tricks on. The model has **34k parameters**, the smallest of the three. Its best mean validation accuracy was 0.817.

- **This is the best of the three PROTEINS models** (GCN⁺ 0.756, GIN⁺ 0.761). It also has the lowest seed variance and is the only one clearly above bag-of-nodes (+1.9 points).
- Even untuned, the plain GPS is already strong (0.770). Tuning adds +0.8 points, within noise.

## 2. Which components matter? (`gps_ablation.png`)

| Variant (5 seeds) | Accuracy | AUC | Δ accuracy |
|---|---|---|---|
| Full GraphGPS | 0.788 | 0.855 | – |
| No normalization | 0.775 | 0.830 | −1.3 |
| No dropout | 0.763 | 0.838 | −2.5 |
| No residual | 0.768 | 0.841 | −2.0 |
| No FFN | 0.793 | 0.860 | +0.5 |
| No positional encoding | 0.788 | 0.845 | 0.0 |
| **No attention (= GIN⁺ block)** | **0.754** | 0.812 | **−3.4** |
| **No local MPNN (pure transformer)** | **0.750** | 0.822 | **−3.8** |
| All tricks off | 0.750 | 0.833 | −3.8 |

- **Both halves of the GPS layer are needed.** Dropping global attention costs 3.4 points and dropping the local GIN costs 3.8. The local branch counts neighbourhood patterns, and attention links distant SSEs in one step.
- Unlike for GCN⁺ and GIN⁺, the classic tricks (dropout, residual, normalization) each help by 1–2.5 points here.
- Positional encoding makes no difference: attention already has global access, so it needs no positional hint.

## 3. What does the model use? (input ablation and feature masking)

| Input | Accuracy |
|---|---|
| Full graph | 0.788 |
| No edges | 0.782 |
| Constant node features | 0.720 |
| SSE type only | 0.755 |

| Feature masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| SSE type (3) | 0.065 | 0.714 |
| SSE length (1) | **0.182** | **0.652** |

- **Removing edges barely matters (−0.5)**: the attention branch recovers the interactions that message passing would provide. Compare GIN⁺, which loses 4.5 points.
- GraphGPS uses the node features most: SSE length especially (masking it costs 13.6 points of accuracy), and SSE type as well (training without length costs 3.2).

## 4. Readout comparison (`gps_readout.png`)

| Readout | Accuracy | AUC | Peak memory | s / epoch |
|---|---|---|---|---|
| **mean (selected)** | **0.788** | 0.855 | 167 MB | 0.37 |
| MinCutPool | 0.759 | 0.838 | 336 MB | 0.46 |
| max | 0.752 | 0.821 | 167 MB | 0.36 |
| add | 0.746 | 0.834 | 167 MB | 0.36 |
| type-aware | 0.745 | 0.819 | 167 MB | 0.40 |
| DiffPool | 0.743 | 0.818 | 516 MB | 0.45 |
| SAGPool | 0.739 ± 0.042 | 0.793 | 167 MB | 0.40 |

- Mean pooling is clearly best for GraphGPS, by 3–5 points over every alternative. Attention has already mixed information across the whole protein, so a simple average is enough. Sum pooling, the best choice for GCN⁺, is worse here.
- The hierarchical pools (SAG, Diff, MinCut) give no benefit, and DiffPool costs 3× the memory.

## 5. Depth and dropout (`gps_depth.png`, `gps_dropout.png`)

| Layers | With residual | Without residual |
|---|---|---|
| 2 | **0.789** | 0.777 |
| 4 | 0.777 | 0.765 |
| 8 | 0.726 | 0.744 |
| 16 | 0.670 | **0.622** |

- Two or three layers is best, and deep GPS degrades sharply, down to 0.62 at 16 layers without residuals.
- Dropout: 0.0 is the worst setting (0.762) and 0.5 the best (0.795). The model benefits from strong regularisation.

## 6. Embeddings (`gps_tsne.png`)

The silhouette score for enzyme vs non-enzyme is 0.106 on raw features and **0.286** on the GraphGPS embedding. That is the best class separation of the three models (GCN⁺ 0.116, GIN⁺ 0.100).

## 7. Robustness, with no retraining (`gps_robustness.png`)

| Damage level | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| Edges removed (fraction) | 0.786 | 0.786 | 0.765 | 0.735 | 0.679 | **0.574** |
| Gaussian feature noise (σ) | 0.786 | 0.792 | 0.798 | 0.792 | – | 0.765 |

- **The most robust of the three to edge loss.** At 50% of edges removed it keeps 0.574, against 0.408 for GCN⁺ and 0.452 for GIN⁺, and 10% loss costs nothing. Attention does not depend on the edges.
- It is insensitive to feature noise.

## 8. Errors by topology (`gps_topology.png`)

| Protein size (SSEs) | Enzyme share | Accuracy |
|---|---|---|
| 5–19 | 0.32 | **0.763** |
| 20–40 | 0.61 | 0.694 |
| 41–226 | 0.87 | 0.895 |

| Clustering coefficient | Accuracy |
|---|---|
| 0.02–0.40 | 0.811 |
| 0.40–0.63 | 0.711 |
| 0.63–0.90 | **0.838** |

- GraphGPS is the best of the three on small proteins (GCN⁺ 0.684, GIN⁺ 0.737) and on highly clustered ones.
- Mid-sized proteins (mixed classes) remain the main source of errors for every model.

## 9. Explainability: GNNExplainer (`gps_explained_proteins.png`)

| Nodes removed | Mean drop in confidence |
|---|---|
| Top 10% SSEs by GNNExplainer | 0.097 |
| Random 10% SSEs | 0.035 |

| Share of SSE type | Helix | Sheet | Turn |
|---|---|---|---|
| All nodes | 0.47 | 0.49 | 0.05 |
| Top-10% nodes | **0.73** | 0.24 | 0.03 |

- **The most faithful explanations of the three:** removing the top-ranked SSEs costs 2.8× as much as removing random ones (GIN⁺ 1.8×, GCN⁺ 1.3×).
- The explainer concentrates on **helices**: 73% of the most important nodes are helices, against 47% of all nodes.

## 10. Efficiency

GraphGPS is the smallest model (34k parameters, against 55k for GIN⁺ and 349k for GCN⁺), but the slowest per epoch: 0.37 s, against 0.14 for GIN⁺ and 0.20 for GCN⁺. That is the cost of dense attention via `to_dense_batch` over proteins of up to 620 nodes. The full notebook took 101 min, against about 45 for the other two.

## 11. Takeaways

1. GraphGPS is the **best PROTEINS model: 0.778 ± 0.016** accuracy (AUC 0.838). It has the smallest parameter count and the lowest variance.
2. **Local message passing and global attention are complementary.** Removing either one costs 3.4–3.8 points.
3. Attention makes the model **robust to missing edges** and gives the clearest embedding (silhouette 0.29).
4. Mean readout is best. Dense hierarchical pools add memory and no accuracy.
5. Its GNNExplainer masks are the most faithful (2.8× random) and point at helices.

## Files

- `gps_proteins_results.json`: all numbers, including the split indices.
- `gps_proteins_search.csv`: the random search.
- `gps_proteins_test_probs.npy`: test probabilities of the 10 final models.
- `graphgps-proteins.log`: Kaggle run log.
- `*.png`: figures referenced above.
