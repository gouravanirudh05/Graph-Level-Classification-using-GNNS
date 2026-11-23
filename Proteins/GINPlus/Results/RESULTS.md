# GIN⁺ on PROTEINS: results and interpretation

Source run: Kaggle `arnavoruganty/gin-plus-proteins` (T4, 42 min). Split: stratified 80/10/10 with `random_state=0` (890 / 111 / 112 graphs). Labels: y=0 is enzyme (59.8%) and y=1 is non-enzyme.

The test set has 112 proteins, so **one graph is worth 0.9 accuracy points**. Differences under about 2 points are within seed noise. Final numbers are the mean ± std over 10 seeds; ablations use 5 seeds and sweeps 3.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.598 | – | – |
| Protein size only | 0.741 | – | – |
| Bag of nodes | 0.759 | – | – |
| Plain GIN (2 layers, no tricks) | 0.732 ± 0.028 | 0.703 ± 0.031 | 0.722 ± 0.060 |
| **GIN⁺ (tuned)** | **0.761 ± 0.025** | **0.744 ± 0.027** | **0.802 ± 0.025** |

The best configuration had hidden width 64, 2 layers, dropout 0.2, mean readout, learning rate 1e-2, no weight decay, and all tricks on. The model has 55k parameters, about a sixth of GCN⁺. Its best mean validation accuracy was 0.829, which is 7 points above test: selecting on a 111-graph validation set is optimistic.

- Tuning gains **+2.9 points of accuracy and +8 points of AUC** over the plain GIN. The plain GIN is the weakest baseline on PROTEINS.
- GIN⁺ ties the bag-of-nodes reference and GCN⁺ (0.756). GraphGPS (0.778) is the only model ahead.

## 2. Which tricks matter? (`ginp_ablation.png`)

| Variant (5 seeds) | Accuracy | AUC | Δ accuracy |
|---|---|---|---|
| Full GIN⁺ | 0.761 | 0.794 | – |
| No normalization | 0.784 | 0.835 | **+2.3** |
| **No dropout** | **0.732** | 0.780 | **−2.9** |
| No residual | 0.752 | 0.792 | −0.9 |
| No FFN | 0.777 | 0.840 | +1.6 |
| No positional encoding | 0.752 | 0.782 | −0.9 |
| All tricks off | 0.755 | 0.760 | −0.5 |

- **Dropout is the trick GIN needs**: its sum aggregation plus MLP overfits quickly on 890 training graphs.
- Batch normalization and the FFN *hurt* here. Removing either one raises both accuracy and AUC. The search tuned width and depth with these tricks always on, so a search that also turned them off might do better. Worth a mention as a limitation.

## 3. What does the model use? (input ablation and feature masking)

| Input | Accuracy |
|---|---|
| Full graph | 0.761 |
| **No edges** | **0.716** |
| Constant node features | 0.718 |
| SSE type only | 0.757 |

| Feature masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| SSE type (3) | 0.022 | 0.696 |
| SSE length (1) | 0.143 | 0.688 |

- **GIN⁺ uses the graph structure more than the other two models do**: removing edges costs 4.5 points (GCN⁺ −2.1, GraphGPS −0.5). This fits GIN's design. With sum aggregation it can count neighbourhood patterns, which a mean-style GCN cannot.
- As with GCN⁺, SSE length carries more of the decision than SSE type.

## 4. Readout comparison (`ginp_readout.png`)

| Readout | Accuracy | AUC | Peak memory | s / epoch |
|---|---|---|---|---|
| **SAGPool** | **0.782** | 0.842 | 51 MB | 0.17 |
| max | 0.773 | 0.841 | 49 MB | 0.13 |
| DiffPool | 0.768 | 0.820 | 436 MB | 0.20 |
| add | 0.764 | 0.827 | 47 MB | 0.14 |
| mean (selected) | 0.761 | 0.794 | 47 MB | 0.14 |
| MinCutPool | 0.730 | 0.802 | 261 MB | 0.23 |
| type-aware | 0.718 | 0.777 | 49 MB | 0.15 |

- The search picked mean pooling on validation, but SAGPool and max are about 1–2 points better on test and about 5 points better on AUC. This is another sign that validation-based selection on 111 graphs is noisy.
- DiffPool uses **9× the memory** of the flat readouts for no gain. MinCutPool and the type-aware readout are the worst.

## 5. Depth and dropout (`ginp_depth.png`, `ginp_dropout.png`)

| Layers | With residual | Without residual |
|---|---|---|
| 2 | 0.759 | **0.774** |
| 4 | 0.732 | 0.771 |
| 8 | 0.732 | 0.702 |
| 16 | 0.670 | 0.661 |

- Shallow is better: 2–4 layers are best and 16 layers loses about 10 points either way. Residuals do not rescue deep GIN on this small dataset.
- Dropout: 0.0 is the worst setting (0.726) and 0.5 the best (0.759). This agrees with the ablation: GIN⁺ needs regularisation.

## 6. Embeddings (`ginp_tsne.png`)

The silhouette score for enzyme vs non-enzyme is 0.106 on raw features and **0.100** on the GIN⁺ embedding. The embedding does not separate the classes better than raw features; they overlap in t-SNE.

## 7. Robustness, with no retraining (`ginp_robustness.png`)

| Damage level | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| Edges removed (fraction) | 0.786 | 0.753 | 0.726 | 0.652 | 0.589 | **0.452** |
| Gaussian feature noise (σ) | 0.786 | 0.783 | 0.786 | 0.765 | – | 0.762 |

- Fragile to edge loss: half the edges gone puts it below the majority baseline. Sum aggregation's output scales with node degree, so losing edges shifts the embeddings.
- Feature noise costs only 2.4 points even at σ = 0.5.

## 8. Errors by topology (`ginp_topology.png`)

| Protein size (SSEs) | Enzyme share | Accuracy |
|---|---|---|
| 5–19 | 0.32 | 0.737 |
| 20–40 | 0.61 | **0.694** |
| 41–226 | 0.87 | 0.921 |

| Clustering coefficient | Accuracy |
|---|---|
| 0.02–0.40 | 0.811 |
| 0.40–0.63 | 0.737 |
| 0.63–0.90 | 0.811 |

- Mid-sized proteins are the hardest, because the classes are most mixed there (61% enzyme).
- Large proteins are easy because most of them are enzymes.

## 9. Explainability: GNNExplainer (`ginp_explained_proteins.png`)

| Nodes removed | Mean drop in confidence |
|---|---|
| Top 10% SSEs by GNNExplainer | 0.102 |
| Random 10% SSEs | 0.057 |

| Share of SSE type | Helix | Sheet | Turn |
|---|---|---|---|
| All nodes | 0.47 | 0.49 | 0.05 |
| Top-10% nodes | **0.59** | 0.36 | 0.05 |

- The explanations are faithful at about 1.8× the random drop, better than GCN⁺ (1.3×).
- Helices are over-represented among the important nodes.

## 10. Takeaways

1. GIN⁺ reaches **0.761 ± 0.025** accuracy (AUC 0.802). That is +2.9 points over the plain GIN, using the smallest model (55k parameters) and the fastest epochs.
2. **Dropout** is the trick that matters for GIN (−2.9 points without it). Batch norm and the FFN hurt on this small dataset.
3. GIN⁺ is the model that **relies most on edges** (−4.5 points without them), and it is fragile when they are removed at test time.
4. SAGPool and max readouts beat the selected mean readout on test. Dense pools are expensive and give nothing.

## Files

- `ginp_proteins_results.json`: all numbers, including the split indices.
- `ginp_proteins_search.csv`: the random search.
- `ginp_proteins_test_probs.npy`: test probabilities of the 10 final models.
- `gin-plus-proteins.log`: Kaggle run log.
- `*.png`: figures referenced above.
