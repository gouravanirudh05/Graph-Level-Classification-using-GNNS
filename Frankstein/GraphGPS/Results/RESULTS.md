# GraphGPS on FRANKENSTEIN: results and interpretation

Source run: Kaggle `gouravanirudh/graphgps-frankenstein` (325 min on GPU). Split: stratified 80/10/10 with `random_state=0` (3,469 / 434 / 434 molecules). One test molecule is 0.23 points. Final numbers are mean ± std over 10 seeds; ablations use 5 seeds and sweeps 3.

## 1. Headline

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.553 | – | – |
| Molecule size only | 0.576 | – | – |
| Bag of atoms | 0.548 | – | – |
| Plain GPS | 0.554 ± 0.009 | 0.418 ± 0.076 | 0.538 ± 0.035 |
| **GraphGPS (tuned)** | 0.733 ± 0.015 | 0.729 ± 0.017 | 0.799 ± 0.017 |

Best configuration: `{'hidden': 32, 'layers': 3, 'dropout': 0.0, 'pool': 'degree', 'lr': 0.01, 'wd': 0.0001, 'heads': 4, 'norm': True, 'res': True, 'ffn': True, 'pe': True}`. Validation accuracy 0.757; 48,325 parameters. The plain GPS stays at the majority rate with the fixed reference settings. *All tricks off*, which keeps the tuned width, depth and readout, is the fairer comparison.

## 2. Which tricks matter?

| Variant | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| full GraphGPS | 0.741 ± 0.010 | 0.738 | 0.806 | +0.0 |
| no normalization | 0.657 ± 0.018 | 0.644 | 0.724 | -8.4 |
| no residual | 0.717 ± 0.016 | 0.714 | 0.783 | -2.4 |
| no FFN | 0.718 ± 0.015 | 0.713 | 0.794 | -2.3 |
| no positional encoding | 0.707 ± 0.006 | 0.705 | 0.769 | -3.5 |
| no attention (= GIN⁺ block) | 0.735 ± 0.019 | 0.728 | 0.804 | -0.7 |
| no local MPNN (pure transformer) | 0.673 ± 0.022 | 0.669 | 0.733 | -6.8 |
| all tricks off | 0.646 ± 0.045 | 0.628 | 0.716 | -9.5 |

## 3. What does the model use?

| Input | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| tuned | 0.741 ± 0.010 | 0.738 | 0.806 | +0.0 |
| no edges (A = I) | 0.550 ± 0.025 | 0.496 | 0.535 | -19.1 |
| constant features | 0.701 ± 0.010 | 0.699 | 0.771 | -4.1 |
| shuffled atoms | 0.690 ± 0.025 | 0.680 | 0.750 | -5.1 |

| Input masked to its training mean | Drop in confidence | Accuracy |
|---|---|---|
| image (780) | 0.202 | 0.641 |
| positional encoding (16) | 0.184 | 0.555 |

## 4. Readouts

| Readout | Accuracy | F1 | AUC | Δ accuracy (points) |
|---|---|---|---|---|
| degree | 0.741 ± 0.010 | 0.738 | 0.806 | +0.0 |
| mean | 0.738 ± 0.012 | 0.733 | 0.797 | -0.3 |
| max | 0.746 ± 0.013 | 0.741 | 0.809 | +0.5 |
| add | 0.724 ± 0.022 | 0.718 | 0.793 | -1.7 |
| sag | 0.720 ± 0.011 | 0.714 | 0.800 | -2.1 |
| diff | 0.707 ± 0.010 | 0.700 | 0.778 | -3.5 |
| mincut | 0.733 ± 0.010 | 0.729 | 0.798 | -0.8 |

| Readout | Parameters | s / epoch | Peak memory (MB) |
|---|---|---|---|
| mean | 46,277 | 1.15 | 293 |
| max | 46,277 | 1.12 | 293 |
| add | 46,277 | 1.14 | 293 |
| degree | 48,325 | 1.21 | 293 |
| sag | 47,334 | 1.25 | 293 |
| diff | 47,498 | 1.37 | 293 |
| mincut | 48,522 | 1.43 | 293 |

## 5. Depth and dropout

| Layers | With residual | Without |
|---|---|---|
| 2 | 0.724 | 0.728 |
| 4 | 0.726 | 0.707 |
| 8 | 0.723 | 0.697 |
| 16 | 0.676 | 0.667 |

| Dropout | Accuracy |
|---|---|
| 0.0 | 0.734 ± 0.014 |
| 0.1 | 0.743 ± 0.015 |
| 0.2 | 0.739 ± 0.005 |
| 0.3 | 0.730 ± 0.014 |
| 0.5 | 0.716 ± 0.009 |

## 6. Embeddings, robustness and topology (selected single model)

Silhouette: mean of raw node features 0.003, tuned GraphGPS embedding 0.054.

| Damage level | Edges removed | Pixel noise σ |
|---|---|---|
| 0.0 | 0.747 | 0.747 |
| 0.1 | 0.644 | 0.717 |
| 0.2 | 0.608 | 0.670 |
| 0.3 | 0.542 | 0.634 |
| 0.4 | 0.506 | – |
| 0.5 | 0.501 | 0.598 |

| Property | Range | Graphs | Mutagenic share | Accuracy |
|---|---|---|---|---|
| size | (2.999, 12.0] | 149 | 0.50 | 0.698 |
| size | (12.0, 19.0] | 144 | 0.62 | 0.750 |
| size | (19.0, 194.0] | 141 | 0.54 | 0.794 |
| degree | (1.332, 2.0] | 216 | 0.45 | 0.755 |
| degree | (2.0, 2.164] | 73 | 0.48 | 0.699 |
| degree | (2.164, 2.476] | 145 | 0.74 | 0.759 |
| rings | (-0.001, 1.0] | 205 | 0.45 | 0.746 |
| rings | (1.0, 3.0] | 145 | 0.60 | 0.752 |
| rings | (3.0, 12.0] | 84 | 0.73 | 0.738 |

## 7. Explainability

| Removed | Mean drop in confidence |
|---|---|
| Top 10% by GNNExplainer | 0.189 |
| Random 10% | 0.138 |

## 8. Takeaways

1. Normalization is the most important trick (−8.4 points without it), followed by the positional encoding (−3.5), residuals (−2.4) and the FFN (−2.3). There is no *no dropout* row because the tuned dropout is 0.
2. Local message passing is needed (−6.8 without it), while removing attention costs only 0.7 points, within the seed spread. Attention is not what makes this model work on molecules this small.
3. The bonds are essential (0.550 without them). Constant features cost 4.1 points, and shuffling the atoms costs 5.1.
4. Max (0.746), degree-aware (0.741) and mean (0.738) are tied. DiffPool is lowest (0.707).
5. Two to eight layers are equally good with residuals; at 16 layers accuracy drops to 0.676 even with them.
6. Bond removal hurts (below the majority rate at 30%). Pixel noise hurts more than for GCN⁺ and GIN⁺ (0.598 at σ = 0.5), and masking image patches has the largest effect near the centre, where the strokes are.
7. GNNExplainer is somewhat faithful: 0.189 drop for its top atoms against 0.138 for random ones.

## Files

`gps_frankenstein_results.json` (all numbers and split indices), `gps_frankenstein_search.csv`, `gps_frankenstein_test_probs.npy`, `graphgps-frankenstein.log`, and the figures (`*.png`).
