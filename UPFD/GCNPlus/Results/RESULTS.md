# GCN⁺ on UPFD Gossipcop: results and interpretation

Source run: Kaggle `gouravanirudh/gcn-plus-upfd` (T4, 84 min). Split: the official UPFD split (1,092 train, 546 validation, 3,826 test). Each test graph is worth about 0.03 accuracy points.

The final numbers are the mean ± std over 10 seeds. Ablations use 5 seeds and sweeps use 3, so their "full model" row differs slightly from the headline number.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.501 | – | – |
| Cascade size only (logistic regression) | 0.752 | – | – |
| News-node features only (logistic regression) | 0.937 | – | – |
| Plain GCN (2 layers, no tricks) | 0.615 ± 0.141 | 0.517 ± 0.216 | 0.910 ± 0.035 |
| **GCN⁺ (tuned)** | **0.970 ± 0.002** | **0.970 ± 0.002** | **0.990 ± 0.003** |

The best configuration had hidden width 128, 4 layers, dropout 0.2, SAGPool readout, learning rate 1e-3, no weight decay, and all tricks on. The model has 400k parameters. Its best mean validation accuracy was 0.975 (`gcnp_upfd_search.csv`, `gcnp_search.png`).

- The tricks turn GCN from an unreliable model into a strong one. The plain GCN has a good AUC (0.91), so it ranks graphs reasonably well. Its accuracy, however, swings from seed to seed (std 0.14): on some seeds it collapses to predicting mostly one class. The normalization, residual and FFN layers fix that training instability.
- GCN⁺ beats the strongest non-graph reference (a classifier on the news node's features) by 3.3 points.
- The permutation check changed outputs by at most 1.8e-7, so the model is invariant to node order, as it should be.

## 2. Which tricks matter? (`gcnp_ablation.png`)

| Variant (5 seeds) | Accuracy | Δ vs full |
|---|---|---|
| Full GCN⁺ | 0.970 | – |
| No normalization | 0.969 | −0.1 |
| No dropout | 0.962 | −0.8 |
| No residual | 0.964 | −0.6 |
| No FFN | 0.960 | −1.0 |
| No positional encoding | 0.973 | +0.3 |
| All tricks off | 0.780 ± 0.255 | −19.1 |

- No single trick is essential, but removing all of them is catastrophic and very unstable. The tricks overlap: each one stabilises training in a similar way.
- FFN, dropout and residuals each give 0.6–1.0 points.
- Random-walk positional encodings do not help on cascades. Removing them is slightly better, although the gain is within about 1 std. Cascades are shallow trees (88% of test cascades have depth ≤ 3), so random-walk statistics carry little information here.

## 3. What does the model actually use? Input ablation and feature masking

| Input | Accuracy |
|---|---|
| Full graph | 0.970 |
| **No edges (each node sees only itself)** | **0.978** |
| Constant node features (structure only) | 0.821 |
| News node only | 0.959 |

| Feature block masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| News node: text (300) | 0.008 | 0.973 |
| News node: profile (10) | 0.001 | 0.973 |
| **Users: text (300)** | **0.439** | **0.501** |
| Users: profile (10) | 0.036 | 0.963 |

These are the most important findings for the presentation:

- **The retweet edges are not needed on Gossipcop.** Deleting all edges gives slightly *higher* accuracy. Pooling the users' features already separates the classes, and message passing adds nothing on top.
- Structure alone still carries signal: with constant features the model reaches 0.82, above the 0.75 of cascade size alone.
- **The decision comes from the users' text embeddings**, i.e. who shared the story. Masking them drops accuracy to chance (0.50). Masking the news node's own text barely matters (−0.8 points of confidence), even though a model trained on the news node alone reaches 0.96. The two signals are redundant, and the trained model relies on the user side.

## 4. Readout comparison (`gcnp_readout.png`)

| Readout | Accuracy | AUC | Peak memory | s / epoch |
|---|---|---|---|---|
| max | **0.973** | 0.992 | 483 MB | 0.28 |
| news-aware (root ‖ mean ‖ max users) | 0.972 | 0.991 | 482 MB | 0.31 |
| SAGPool (selected) | 0.970 | 0.990 | 482 MB | 0.34 |
| mean | 0.968 | 0.990 | 480 MB | 0.29 |
| add | 0.962 | 0.987 | 480 MB | 0.29 |
| DiffPool | 0.961 | 0.984 | 555 MB | 0.37 |
| MinCutPool | 0.961 | 0.985 | 520 MB | 0.40 |

- All readouts land within 1.3 points of each other. Max, news-aware, SAGPool and mean are tied within about 1 std.
- The search picked SAGPool on validation, but max pooling is marginally better on test.
- The dense hierarchical pools (DiffPool, MinCutPool) are the worst, use 8–15% more memory and train up to 40% slower than the flat readouts. Cascades are small, star-like trees, so hierarchical clustering has nothing to discover.
- The news-aware readout's advantage is small, which fits section 3: the root node is not where the signal lives.

## 5. Depth and dropout (`gcnp_depth.png`, `gcnp_dropout.png`)

| Layers | With residual | Without residual |
|---|---|---|
| 2 | 0.972 | 0.974 |
| 4 | 0.966 | 0.964 |
| 8 | 0.972 | 0.963 |
| 16 | **0.974** | **0.939 ± 0.043** |

- Residual connections keep a 16-layer GCN⁺ as accurate as a 2-layer one. Without them, accuracy falls and becomes unstable at depth, the usual over-smoothing / optimisation effect.
- Depth itself buys nothing, consistent with edges not being needed.
- Dropout: 0.0 is the worst setting (0.964), and anything in 0.1–0.5 gives 0.967–0.973. Some dropout is needed, but the exact value matters little.

## 6. Embeddings (`gcnp_tsne.png`)

The silhouette score for fake vs real rises from 0.149 (mean of raw node features) to **0.515** (GCN⁺ graph embedding). The learned embedding separates the classes far better than the raw features.

## 7. Robustness, with no retraining (`gcnp_robustness.png`)

| Damage level | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| Edges removed (fraction) | 0.973 | 0.964 | 0.958 | 0.951 | 0.936 |
| Gaussian feature noise (σ) | 0.973 | 0.969 | 0.941 | 0.866 | 0.741 |

- The model degrades gracefully when edges are lost: half the edges gone costs only 3.6 points.
- It is much more sensitive to feature noise. This matches sections 3 and 4: the model reads node features, not structure.

## 8. Errors by topology (`gcnp_topology.png`)

| Cascade size (nodes) | Test graphs | Fake share | Accuracy |
|---|---|---|---|
| 3–19 | 982 | 0.89 | **0.955** |
| 20–49 | 940 | 0.58 | 0.981 |
| 50–83 | 951 | 0.24 | 0.981 |
| 84–199 | 953 | 0.27 | 0.974 |

| Star-ness (share of users who retweet the news directly) | Accuracy |
|---|---|
| 0.03–0.67 | **0.946** |
| 0.67–0.85 | 0.979 |
| 0.85–0.95 | 0.980 |
| 0.95–0.99 | 0.985 |

- Errors concentrate in **small cascades** and in **deep, chain-like cascades** (low star-ness). Small cascades have few users to pool over, so the user-text signal is thinner.
- Fake news dominates small cascades (89%), which explains why size alone already scores 0.75.
- Cascade depth (≤ 3 vs > 3 hops) makes little difference (0.973 vs 0.969).

## 9. Explainability: GNNExplainer (`gcnp_explained_cascades.png`)

The test used 100 test cascades with at least 20 nodes. For each, it removed the 10% of users that GNNExplainer rated most important and measured the drop in confidence:

| Users removed | Mean drop in confidence |
|---|---|
| Top 10% by GNNExplainer | 0.0009 |
| Random 10% | 0.0083 |

- **The node explanations are not faithful on UPFD.** Removing the "most important" users hurts *less* than removing random users.
- On average, 34 nodes are ranked above the news node.
- This fits the rest of the analysis. Because the model aggregates user features redundantly and ignores edges, no small set of users is decisive. The informative explanation is the feature-block masking in section 3: user text is the deciding input.

## 10. Takeaways for the presentation

1. GCN⁺ reaches **0.970 ± 0.002** accuracy (AUC 0.990), up from 0.615 for a plain GCN. The Luo et al. tricks matter mainly as a group, for training stability.
2. On Gossipcop, the label is predictable from **who shared the story** (user text embeddings). The cascade's edges are unnecessary: removing them even helps slightly. Keep this in mind when comparing models; GraphGPS's attention should not be expected to add much.
3. Simple readouts (max, mean, news-aware) are as good as or better than SAGPool, DiffPool and MinCutPool. The dense pools cost extra memory and time for nothing.
4. Errors are concentrated in small and chain-like cascades.
5. GNNExplainer node masks fail the deletion-fidelity test here. Feature-level masking is the explanation to report.

## Files

- `gcnp_upfd_results.json`: all numbers above.
- `gcnp_upfd_search.csv`: the 30-trial random search.
- `gcnp_upfd_test_probs.npy`: test probabilities of the 10 final models.
- `gcn-plus-upfd.log`: Kaggle run log.
- `*.png`: figures referenced above.
