# GIN⁺ on FRANKENSTEIN: results and interpretation

Source run: Kaggle `saisathviksvn/gin-plus-frankenstein` (171 min on GPU). Split: stratified 80/10/10 with `random_state=0` (3,469 / 434 / 434 molecules). One test molecule is 0.23 points. Final numbers are mean ± std over 10 seeds; ablations use 5 seeds and sweeps 3.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.553 | – | – |
| Molecule size only | 0.576 | – | – |
| Bag of atoms | 0.548 | – | – |
| Plain GIN | 0.653 ± 0.022 | 0.645 ± 0.020 | 0.701 ± 0.019 |
| **GIN⁺ (tuned)** | 0.720 ± 0.020 | 0.713 ± 0.022 | 0.792 ± 0.015 |

Best configuration: `{'hidden': 128, 'layers': 3, 'dropout': 0.2, 'pool': 'degree', 'lr': 0.001, 'wd': 0.0, 'norm': True, 'res': True, 'ffn': True, 'pe': True}`. Validation accuracy 0.756; 436,453 parameters. The plain GIN already reaches 0.653, unlike the plain GCN and plain GPS, which stay at the majority rate.

## 2. Which tricks matter?

| Variant | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| full GIN⁺ | 0.717 ± 0.022 | 0.711 | 0.793 | +0.0 |
| no normalization | 0.729 ± 0.014 | 0.724 | 0.790 | +1.2 |
| no dropout | 0.719 ± 0.008 | 0.715 | 0.779 | +0.2 |
| no residual | 0.696 ± 0.023 | 0.691 | 0.767 | -2.0 |
| no FFN | 0.707 ± 0.023 | 0.701 | 0.777 | -1.0 |
| no positional encoding | 0.686 ± 0.021 | 0.675 | 0.742 | -3.1 |
| all tricks off | 0.666 ± 0.022 | 0.651 | 0.724 | -5.1 |

## 3. What does the model use?

| Input | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| tuned | 0.717 ± 0.022 | 0.711 | 0.793 | +0.0 |
| no edges (A = I) | 0.572 ± 0.011 | 0.539 | 0.569 | -14.4 |
| constant features | 0.724 ± 0.007 | 0.720 | 0.785 | +0.7 |
| shuffled atoms | 0.694 ± 0.015 | 0.685 | 0.746 | -2.3 |

| Input masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| image (780) | 0.162 | 0.613 |
| positional encoding (16) | 0.160 | 0.555 |

## 4. Readouts

| Readout | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| degree | 0.717 ± 0.022 | 0.711 | 0.793 | +0.0 |
| mean | 0.714 ± 0.014 | 0.703 | 0.779 | -0.3 |
| max | 0.719 ± 0.009 | 0.709 | 0.789 | +0.2 |
| add | 0.713 ± 0.005 | 0.706 | 0.782 | -0.3 |
| sag | 0.688 ± 0.019 | 0.685 | 0.747 | -2.9 |
| diff | 0.706 ± 0.016 | 0.702 | 0.771 | -1.1 |
| mincut | 0.728 ± 0.012 | 0.725 | 0.791 | +1.1 |

| Readout | Parameters | s / epoch | Peak memory (MB) |
|---|---|---|---|
| mean | 403,685 | 0.72 | 136 |
| max | 403,685 | 0.74 | 137 |
| add | 403,685 | 0.71 | 136 |
| degree | 436,453 | 0.84 | 136 |
| sag | 420,198 | 0.88 | 136 |
| diff | 420,842 | 1.02 | 221 |
| mincut | 437,226 | 1.06 | 222 |

## 5. Depth and dropout

| Layers | With residual | Without |
|---|---|---|
| 2 | 0.717 | 0.705 |
| 4 | 0.707 | 0.727 |
| 8 | 0.705 | 0.694 |
| 16 | 0.722 | 0.684 |

| Dropout | Accuracy |
|---|---|
| 0.0 | 0.726 ± 0.024 |
| 0.1 | 0.717 ± 0.026 |
| 0.2 | 0.720 ± 0.012 |
| 0.3 | 0.733 ± 0.005 |
| 0.5 | 0.729 ± 0.034 |

## 6. Embeddings, robustness and topology (selected single model)

Silhouette: mean of raw node features 0.003, tuned GIN⁺ embedding 0.028.

| Damage level | Edges removed | Pixel noise σ |
|---|---|---|
| 0.0 | 0.730 | 0.730 |
| 0.1 | 0.633 | 0.728 |
| 0.2 | 0.575 | 0.723 |
| 0.3 | 0.508 | 0.708 |
| 0.4 | 0.479 | – |
| 0.5 | 0.462 | 0.667 |

| Property | Range | Graphs | Mutagenic share | Accuracy |
|---|---|---|---|---|
| size | (2.999, 12.0] | 149 | 0.50 | 0.678 |
| size | (12.0, 19.0] | 144 | 0.62 | 0.778 |
| size | (19.0, 194.0] | 141 | 0.54 | 0.738 |
| degree | (1.332, 2.0] | 216 | 0.45 | 0.718 |
| degree | (2.0, 2.164] | 73 | 0.48 | 0.740 |
| degree | (2.164, 2.476] | 145 | 0.74 | 0.745 |
| rings | (-0.001, 1.0] | 205 | 0.45 | 0.717 |
| rings | (1.0, 3.0] | 145 | 0.60 | 0.724 |
| rings | (3.0, 12.0] | 84 | 0.73 | 0.774 |

## 7. Explainability

| Removed | Mean drop in confidence |
|---|---|
| Top 10% by GNNExplainer | 0.155 |
| Random 10% | 0.119 |

## 8. Takeaways

1. Positional encoding (−3.1 points without it) and residuals (−2.0) help most. The FFN is worth about 1 point, and normalization and dropout do not help on the test split; all single-trick changes except the positional encoding are near or below the seed spread of about 2 points.
2. The bonds are essential (0.572 without them). Constant features score 0.724 against 0.717 for the full input, so this GIN⁺ gains nothing measurable from the images.
3. Six of the seven readouts lie within 0.706 to 0.728 and cannot be separated reliably: MinCutPool 0.728, max 0.719, degree-aware 0.717. Only SAGPool (0.688) falls behind.
4. Residuals matter once the network is deep: at 16 layers 0.722 with them and 0.684 without.
5. Random bond removal is very damaging (below the majority rate at 30%), while pixel noise up to σ = 0.2 costs under 1 point.
6. GNNExplainer is only weakly faithful: 0.155 drop for its top atoms against 0.120 for random ones. It shows no preference for chain ends, rings or branches.

## Files

`ginp_frankenstein_results.json` (all numbers and split indices), `ginp_frankenstein_search.csv`, `ginp_frankenstein_test_probs.npy`, `gin-plus-frankenstein.log`, and the figures (`*.png`).
