# Graph-Level Classification with GNNs

Course project (GNN, Semester 7). We compare three graph neural networks on three binary graph-classification benchmarks. Each model is tuned separately on each dataset. We then analyse where the models work, where they fail, and why.

**Team:** Gourav Anirudh (IMT2023005) · SVN Sathvik (IMT2023001) · Arnav Oruganty (IMT2023078)

## Headline results

Test accuracy, mean ± std over 10 seeds:

| Dataset | GCN | GIN | GraphGPS | Strongest simple baseline |
|---|---|---|---|---|
| UPFD Gossipcop | 0.970 ± 0.002 | **0.972 ± 0.003** | 0.971 ± 0.005 | 0.937 (news-node features) |
| PROTEINS | 0.756 ± 0.022 | 0.761 ± 0.025 | **0.778 ± 0.016** | 0.759 (bag of nodes) |
| FRANKENSTEIN | 0.603 ± 0.014 | 0.688 ± 0.020 | **0.742 ± 0.018** | 0.576 (molecule size) |

UPFD and PROTEINS use the enhanced models GCN⁺ and GIN⁺ (see [Models](#models)). FRANKENSTEIN uses the earlier plain models; see [Limitations](#limitations).

**Main findings**
- **GraphGPS wins where structure carries the signal and only ties where it doesn't.** It leads by 5.4 points on FRANKENSTEIN and 1.7 on PROTEINS, and ties on UPFD.
- **On UPFD the graph is almost decoration.** All three models score slightly *higher* with every retweet edge removed. Masking the users' text embeddings drops GCN⁺ and GraphGPS to chance. The cheapest model, a 1-layer GIN⁺ with 12.8k parameters, matches the others.
- **On FRANKENSTEIN the bonds are the signal.** GraphGPS loses 18 points without them.
- **On PROTEINS both halves of GraphGPS are needed.** Removing either the local GIN branch or global attention costs 3–4 points. Attention also makes GraphGPS the most robust model to missing edges, and gives it the most faithful explanations.
- **The Luo et al. tricks matter mostly as a group.** Removing any single trick rarely costs more than a point. Removing all of them costs about 20 points on UPFD for GCN⁺ and GraphGPS, and GraphGPS cannot train without BatchNorm.
- **Dense hierarchical pooling (DiffPool, MinCutPool) never pays off.** It costs 3–9× the memory of a flat readout with no accuracy gain. Plain max pooling is the best UPFD readout for every model.

## Datasets

| Dataset | Graphs | Node features | Avg nodes / edges | Task | Split |
|---|---|---|---|---|---|
| [UPFD](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.UPFD.html) Gossipcop | 5,464 | 310 (spaCy text + profile) | 58 / 56.5 | fake vs real news cascade | official 1,092 / 546 / 3,826 |
| [PROTEINS](https://chrsmrrs.github.io/datasets/) | 1,113 | 4 (SSE type one-hot + length) | 39 / 73 | enzyme vs non-enzyme | stratified 80/10/10, seed 0 |
| [FRANKENSTEIN](https://chrsmrrs.github.io/datasets/) | 4,337 | 780 (MNIST pixels per atom) | 16.9 / 17.9 | mutagenic vs not | stratified 80/10/10, seed 0 |

Every dataset loads through PyTorch Geometric, and the notebooks download it automatically. UPFD cascades are directed trees rooted at the news node, so they are made undirected before computing positional encodings. In PROTEINS, label 0 is *enzyme* (663 graphs) and label 1 is *non-enzyme* (450).

## Models

| Model | Message passing | Notes |
|---|---|---|
| **GCN⁺** | `GCNConv`, a degree-normalised mean | Cheapest; least expressive |
| **GIN⁺** | `GINConv`, a sum through an MLP | As expressive as the 1-WL test (Xu et al., 2019) |
| **GraphGPS** | local `GINConv` + global multi-head attention, summed per layer | Rampášek et al., 2022 |

The ⁺ models follow *Can Classic GNNs Be Strong Baselines for Graph-level Tasks?* (Luo, Shi & Wu, ICML 2025). Each block is conv → BatchNorm → ReLU → dropout → residual → feed-forward network, with random-walk positional encodings at the input. Every trick has an on/off switch, so it can be ablated.

**Readout is a hyperparameter.** The search chooses among seven readouts: mean, sum, max, a role-aware readout, SAGPool, DiffPool and MinCutPool. The role-aware readout is [root ‖ mean ‖ max of users] on UPFD, and the mean per SSE type on PROTEINS. Our SAGPool ranks nodes on the raw score, which keeps it exactly permutation-invariant.

## Protocol

- **Search:** 30 random configurations × 3 seeds (20 for FRANKENSTEIN), ranked by mean validation accuracy. The space covers width, depth, dropout, learning rate, weight decay, readout, and attention heads for GraphGPS.
- **Final test:** the best configuration is retrained on 10 fresh seeds and reported as mean ± std of accuracy, macro-F1 and ROC-AUC.
- **Training:** Adam, early stopping on validation accuracy, gradient-norm clipping at 5.
- **Hardware:** every notebook ran end-to-end on a Kaggle T4 GPU, in 42–170 minutes.

### Analyses (Milestone 3)

Each UPFD and PROTEINS notebook runs every analysis below for its model:

| Analysis | What we measure |
|---|---|
| Ablation | each trick switched off; inputs removed (no edges, constant features, news node only / SSE type only) |
| Readout comparison | all 7 readouts with the tuned model: accuracy, AUC, memory, time |
| Depth and dropout | 2–16 layers with and without residuals; dropout sweep |
| Visualisation | t-SNE of graph embeddings, silhouette score vs raw features |
| Efficiency | parameters, seconds per epoch, inference time, peak GPU memory |
| Robustness | random edge removal and Gaussian feature noise at test time |
| Error analysis by topology | accuracy binned by graph size, star-ness or clustering, and depth |
| Explainability | GNNExplainer node masks with deletion fidelity; feature-block masking |

The brief's node-level examples are adapted to graph level. For instance, "accuracy by node degree" becomes accuracy by cascade size and shape.

## Repository layout

```
UPFD/
  GCNPlus/   gcn_plus_upfd.ipynb       Results/
  GINPlus/   gin_plus_upfd.ipynb       Results/
  GraphGPS/  graphgps_upfd.ipynb       Results/
Proteins/
  GCNPlus/   gcn_plus_proteins.ipynb   Results/
  GINPlus/   gin_plus_proteins.ipynb   Results/
  GraphGPS/  graphgps_proteins.ipynb   Results/
Frankstein/
  GCN/               gcn_frankenstein.ipynb                 intermediates/  results/
  GIN/               gin_frankenstein.ipynb                 intermediates/  results/
  GraphTransformer/  graph_transformer_frankenstein.ipynb   intermediates/  results/
```

Each `Results/` folder holds:
- `RESULTS.md`: every number with its interpretation, compared against the other models. **Start here.**
- `*_results.json`: all metrics, the search table and the split indices.
- `*_search.csv`: the random-search trials.
- `*_test_probs.npy`: test probabilities of the 10 final models.
- `*.png`: the figures.

## Running the notebooks

The notebooks are self-contained, and each code cell comes with a markdown cell explaining it.

1. Open a notebook on Kaggle or Colab with a GPU (T4 or better) and internet access.
2. Run all cells. The first cell installs PyTorch Geometric (`pip install torch_geometric`), and the datasets download to `/tmp`.
3. Outputs (JSON, CSV, NPY, PNG) are written to the working directory.

Runtime on a T4: 42–101 minutes for PROTEINS, 75–170 minutes for UPFD, and 52–121 minutes for FRANKENSTEIN, plus about 3–4.5 hours for its pooling study. To smoke-test on CPU, lower the budget in the "experiment budget" cell near the top: epochs, trials and seeds.

## Limitations

- **FRANKENSTEIN used the first pipeline:**
  - plain GCN, GIN and GraphGPS with a BatchNorm switch, not the full five-trick ⁺ models;
  - a 20-configuration search;
  - a separate study of 11 poolings, including attention, CLS-token and PMA pooling.
  Its trick ablations are therefore not directly comparable with UPFD and PROTEINS.
- **PROTEINS has only 112 test graphs,** so one graph is worth 0.9 accuracy points. Differences under about 2 points are within seed noise. Best validation accuracy overstates test accuracy by 4–7 points.
- **The search keeps the tricks on and does not toggle them.** On PROTEINS, the ablations show that BatchNorm and the FFN hurt GIN⁺, so a search over the switches could do better.

## References

- Luo, Shi, Wu. *Can Classic GNNs Be Strong Baselines for Graph-level Tasks?* ICML 2025.
- Kipf, Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
- Xu, Hu, Leskovec, Jegelka. *How Powerful are Graph Neural Networks?* ICLR 2019.
- Rampášek et al. *Recipe for a General, Powerful, Scalable Graph Transformer.* NeurIPS 2022.
- Dou et al. *User Preference-aware Fake News Detection.* SIGIR 2021 (UPFD).
- Borgwardt et al. *Protein Function Prediction via Graph Kernels.* Bioinformatics 2005 (PROTEINS).
- Orsini, Frasconi, De Raedt. *Graph Invariant Kernels.* IJCAI 2015 (FRANKENSTEIN).
- Ying et al. *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.
- Lee, Lee, Kang. *Self-Attention Graph Pooling.* ICML 2019 · Ying et al. *DiffPool.* NeurIPS 2018 · Bianchi et al. *MinCutPool.* ICML 2020.
