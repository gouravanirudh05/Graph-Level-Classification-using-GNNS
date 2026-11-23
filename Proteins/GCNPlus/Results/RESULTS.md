# GCN⁺ on PROTEINS: results and interpretation

Source run: Kaggle `gourasri/gcn-plus-proteins` (T4, 46 min). Split: stratified 80/10/10 with `random_state=0` (890 / 111 / 112 graphs). Labels: y=0 is enzyme (59.8%) and y=1 is non-enzyme.

The test set has only 112 proteins, so **one graph is worth 0.9 accuracy points**. Differences under about 2 points are within seed noise. Final numbers are the mean ± std over 10 seeds; ablations use 5 seeds and sweeps 3.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.598 | – | – |
| Protein size only | 0.741 | – | – |
| Bag of nodes (counts and lengths per SSE type) | 0.759 | – | – |
| Plain GCN (2 layers, no tricks) | 0.764 ± 0.017 | 0.741 ± 0.020 | 0.792 ± 0.008 |
| **GCN⁺ (tuned)** | **0.756 ± 0.022** | **0.734 ± 0.026** | **0.834 ± 0.018** |

The best configuration had hidden width 128, 4 layers, dropout 0, sum (add) readout, learning rate 1e-2, weight decay 1e-5, and all tricks on. The model has 349k parameters. Its best mean validation accuracy was 0.820, which is 6 points above test: picking 1 of 30 configurations on a 111-graph validation set is optimistic.

- **Tuning did not beat the plain GCN on accuracy**; the two are tied within noise. GCN⁺ does improve AUC by 4 points, so its probability ranking is better.
- Neither model beats the bag-of-nodes reference by a meaningful margin. Most of the signal is in how many SSEs of each type a protein has, which matches the literature: GNNs on PROTEINS score about 0.72–0.77.

## 2. Which tricks matter? (`gcnp_ablation.png`)

| Variant (5 seeds) | Accuracy | AUC | Δ accuracy |
|---|---|---|---|
| Full GCN⁺ | 0.761 | 0.841 | – |
| No normalization | 0.763 | 0.849 | +0.2 |
| No residual | 0.754 | 0.822 | −0.7 |
| No FFN | 0.775 | 0.858 | +1.4 |
| **No positional encoding** | **0.734** | 0.806 | **−2.7** |
| All tricks off | 0.775 | 0.852 | +1.4 |

There is no "no dropout" row because the tuned dropout is already 0.

- **Positional encoding is the only trick that clearly helps**, by 2.7 points. Random-walk return probabilities describe the local ring and cluster structure of the protein graph, which a degree-normalised GCN cannot see on its own.
- The other tricks are neutral or slightly harmful on a dataset this small. The FFN adds parameters that overfit.

## 3. What does the model use? (input ablation and feature masking)

| Input | Accuracy |
|---|---|
| Full graph | 0.761 |
| No edges (each node sees only itself) | 0.739 |
| Constant node features (structure only) | 0.702 |
| SSE type only (length removed) | 0.768 |

| Feature masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| SSE type (3) | 0.018 | 0.732 |
| SSE length (1) | 0.095 | 0.741 |

- Edges help a little (+2.1 points), and node features matter more (+5.9 points).
- The trained model leans on SSE length more than on SSE type. A model trained without length does just as well, however, so the two features are largely redundant.

## 4. Readout comparison (`gcnp_readout.png`)

| Readout | Accuracy | AUC | Peak memory | s / epoch |
|---|---|---|---|---|
| **add (selected)** | **0.761** | 0.841 | 110 MB | 0.20 |
| DiffPool | 0.757 | 0.826 | 502 MB | 0.27 |
| mean | 0.755 | 0.796 | 110 MB | 0.19 |
| max | 0.752 | 0.812 | 110 MB | 0.19 |
| SAGPool | 0.729 | 0.802 | 105 MB | 0.22 |
| type-aware (mean per SSE type) | 0.727 | 0.779 | 112 MB | 0.21 |
| MinCutPool | 0.675 ± 0.056 | 0.743 | 337 MB | 0.28 |

- Sum pooling wins. It keeps the protein's size, and size is predictive on its own (0.741).
- DiffPool matches it but needs **4.5× the memory**. MinCutPool is clearly worse and unstable.
- The type-aware readout does not help: averaging per type throws away the counts that make bag-of-nodes work.

## 5. Depth and dropout (`gcnp_depth.png`, `gcnp_dropout.png`)

| Layers | With residual | Without residual |
|---|---|---|
| 2 | **0.786** | 0.771 |
| 4 | 0.768 | 0.762 |
| 8 | 0.750 | 0.729 |
| 16 | 0.732 | 0.741 |

- Shallow is better: 2 layers beats the tuned 4. Residuals help a little at 2–8 layers.
- Dropout falls steadily from 0.762 at 0.0 to 0.705 at 0.5. Unlike on UPFD, this model prefers no dropout.

## 6. Embeddings (`gcnp_tsne.png`)

The silhouette score for enzyme vs non-enzyme is 0.106 on raw features and **0.116** on the GCN⁺ embedding. The two classes overlap heavily in embedding space, consistent with a ceiling of about 0.76 accuracy.

## 7. Robustness, with no retraining (`gcnp_robustness.png`)

| Damage level | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| Edges removed (fraction) | 0.750 | 0.735 | 0.682 | 0.557 | 0.470 | **0.408** |
| Gaussian feature noise (σ) | 0.750 | 0.762 | 0.765 | 0.768 | – | 0.768 |

- **Very fragile to edge loss.** With half the edges gone, accuracy falls below the majority-class baseline. Sum pooling over degree-normalised messages shifts the scale of the graph embedding when the degrees change.
- Feature noise has no effect.

## 8. Errors by topology (`gcnp_topology.png`)

| Protein size (SSEs) | Test graphs | Enzyme share | Accuracy |
|---|---|---|---|
| 5–19 | 38 | 0.32 | 0.684 |
| 20–40 | 36 | 0.61 | 0.667 |
| 41–226 | 38 | 0.87 | **0.895** |

| Clustering coefficient | Accuracy |
|---|---|
| 0.02–0.40 | 0.811 |
| 0.40–0.63 | 0.684 |
| 0.63–0.90 | 0.757 |

- Large proteins are easy (0.89) because 87% of them are enzymes.
- The errors are among small and mid-sized proteins, where the classes are mixed.

## 9. Explainability: GNNExplainer (`gcnp_explained_proteins.png`)

| Nodes removed | Mean drop in confidence |
|---|---|
| Top 10% SSEs by GNNExplainer | 0.178 |
| Random 10% SSEs | 0.135 |

| Share of SSE type | Helix | Sheet | Turn |
|---|---|---|---|
| All nodes | 0.47 | 0.49 | 0.05 |
| Top-10% nodes | **0.66** | 0.34 | 0.00 |

- The explanations are only slightly more faithful than random (1.3× the random drop).
- The explainer does favour **helices**: 66% of top nodes are helices, against 47% overall.

## 10. Takeaways

1. GCN⁺ reaches **0.756 ± 0.022** accuracy (AUC 0.834). It ties the plain GCN and the bag-of-nodes reference on accuracy and is better on AUC.
2. Of the Luo et al. tricks, only **positional encoding** helps GCN on PROTEINS (+2.7). The rest are neutral at this dataset size.
3. **Sum pooling** is best because protein size is informative. Dense pools cost 3–4.5× the memory with no gain.
4. The model is **fragile to missing edges** (0.41 at 50% edge loss) but immune to feature noise.
5. Explanations point at helices but are weakly faithful.

## Files

- `gcnp_proteins_results.json`: all numbers, including the split indices.
- `gcnp_proteins_search.csv`: the random search.
- `gcnp_proteins_test_probs.npy`: test probabilities of the 10 final models.
- `gcn-plus-proteins.log`: Kaggle run log.
- `*.png`: figures referenced above.
