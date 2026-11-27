# GCN⁺ on FRANKENSTEIN: results and interpretation

Source run: Kaggle `saisathviksvn/gcn-plus-frankenstein` (148 min on GPU). Split: stratified 80/10/10 with `random_state=0` (3,469 / 434 / 434 molecules). One test molecule is 0.23 points. Final numbers are mean ± std over 10 seeds; ablations use 5 seeds and sweeps 3.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.553 | – | – |
| Molecule size only | 0.576 | – | – |
| Bag of atoms | 0.548 | – | – |
| Plain GCN | 0.553 ± 0.000 | 0.356 ± 0.000 | 0.485 ± 0.039 |
| **GCN⁺ (tuned)** | 0.730 ± 0.017 | 0.722 ± 0.020 | 0.787 ± 0.016 |

Best configuration: `{'hidden': 128, 'layers': 3, 'dropout': 0.2, 'pool': 'degree', 'lr': 0.001, 'wd': 0.0, 'norm': True, 'res': True, 'ffn': True, 'pe': True}`. Validation accuracy 0.752; 386,146 parameters. The plain GCN predicts the larger class on all ten seeds, so it learned nothing with the fixed reference settings. The gain over it overstates the tricks; *all tricks off*, which keeps the tuned width, depth and readout, is the fairer comparison.

## 2. Which tricks matter?

| Variant | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| full GCN⁺ | 0.732 ± 0.022 | 0.725 | 0.796 | +0.0 |
| no normalization | 0.734 ± 0.008 | 0.728 | 0.813 | +0.1 |
| no dropout | 0.726 ± 0.021 | 0.717 | 0.795 | -0.7 |
| no residual | 0.707 ± 0.019 | 0.702 | 0.781 | -2.5 |
| no FFN | 0.709 ± 0.019 | 0.699 | 0.774 | -2.3 |
| no positional encoding | 0.672 ± 0.021 | 0.660 | 0.726 | -6.0 |
| all tricks off | 0.646 ± 0.010 | 0.640 | 0.711 | -8.6 |

## 3. What does the model use?

| Input | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| tuned | 0.732 ± 0.022 | 0.725 | 0.796 | +0.0 |
| no edges (A = I) | 0.570 ± 0.013 | 0.545 | 0.560 | -16.3 |
| constant features | 0.700 ± 0.030 | 0.695 | 0.758 | -3.3 |
| shuffled atoms | 0.682 ± 0.017 | 0.673 | 0.741 | -5.1 |

| Input masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| image (780) | 0.123 | 0.627 |
| positional encoding (16) | 0.134 | 0.618 |

## 4. Readouts

| Readout | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| degree | 0.732 ± 0.022 | 0.725 | 0.796 | +0.0 |
| mean | 0.708 ± 0.012 | 0.706 | 0.777 | -2.4 |
| max | 0.708 ± 0.029 | 0.704 | 0.780 | -2.4 |
| add | 0.707 ± 0.016 | 0.702 | 0.778 | -2.5 |
| sag | 0.688 ± 0.017 | 0.677 | 0.741 | -4.4 |
| diff | 0.707 ± 0.024 | 0.702 | 0.771 | -2.5 |
| mincut | 0.720 ± 0.004 | 0.713 | 0.792 | -1.2 |

| Readout | Parameters | s / epoch | Peak memory (MB) |
|---|---|---|---|
| mean | 353,378 | 0.68 | 126 |
| max | 353,378 | 0.69 | 127 |
| add | 353,378 | 0.69 | 126 |
| degree | 386,146 | 0.80 | 127 |
| sag | 369,891 | 0.82 | 129 |
| diff | 370,535 | 0.96 | 217 |
| mincut | 386,919 | 1.02 | 218 |

## 5. Depth and dropout

| Layers | With residual | Without |
|---|---|---|
| 2 | 0.724 | 0.712 |
| 4 | 0.720 | 0.705 |
| 8 | 0.725 | 0.707 |
| 16 | 0.723 | 0.709 |

| Dropout | Accuracy |
|---|---|
| 0.0 | 0.725 ± 0.035 |
| 0.1 | 0.728 ± 0.026 |
| 0.2 | 0.743 ± 0.021 |
| 0.3 | 0.722 ± 0.033 |
| 0.5 | 0.710 ± 0.004 |

## 6. Embeddings, robustness and topology (selected single model)

Silhouette: mean of raw node features 0.003, tuned GCN⁺ embedding 0.024.

| Damage level | Edges removed | Pixel noise σ |
|---|---|---|
| 0.0 | 0.763 | 0.763 |
| 0.1 | 0.627 | 0.758 |
| 0.2 | 0.574 | 0.756 |
| 0.3 | 0.502 | 0.734 |
| 0.4 | 0.470 | – |
| 0.5 | 0.458 | 0.676 |

| Property | Range | Graphs | Mutagenic share | Accuracy |
|---|---|---|---|---|
| size | (2.999, 12.0] | 149 | 0.50 | 0.745 |
| size | (12.0, 19.0] | 144 | 0.62 | 0.785 |
| size | (19.0, 194.0] | 141 | 0.54 | 0.759 |
| degree | (1.332, 2.0] | 216 | 0.45 | 0.773 |
| degree | (2.0, 2.164] | 73 | 0.48 | 0.740 |
| degree | (2.164, 2.476] | 145 | 0.74 | 0.759 |
| rings | (-0.001, 1.0] | 205 | 0.45 | 0.771 |
| rings | (1.0, 3.0] | 145 | 0.60 | 0.738 |
| rings | (3.0, 12.0] | 84 | 0.73 | 0.786 |

## 7. Explainability

| Removed | Mean drop in confidence |
|---|---|
| Top 10% by GNNExplainer | 0.135 |
| Random 10% | 0.141 |

## 8. Takeaways

1. Positional encoding is the most important trick (−6.0 points without it). Removing residuals costs 2.5 points and removing the FFN 2.3. Normalization and dropout make no clear difference.
2. The bonds are essential: without them accuracy is 0.570. With constant features it is still 0.700, so the skeleton carries most of the signal and the images add about 3 points.
3. The degree-aware readout scores highest (0.732), ahead of MinCutPool (0.720), and SAGPool is lowest (0.688). The gap between the first two is within the seed spread.
4. Depth from 2 to 16 layers changes little. Residuals are ahead by 1 to 2 points at every depth.
5. Random bond removal is very damaging: at 20% removed the model is near the majority rate, and at 30% below it. Pixel noise is tolerated up to about σ = 0.2.
6. GNNExplainer is not faithful here: removing its top atoms lowers confidence no more than removing random atoms (0.135 against 0.141).

## Files

`gcnp_frankenstein_results.json` (all numbers and split indices), `gcnp_frankenstein_search.csv`, `gcnp_frankenstein_test_probs.npy`, `gcn-plus-frankenstein.log`, and the figures (`*.png`).
