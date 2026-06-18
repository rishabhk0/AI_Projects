# TacticalGAT — Football Tactical Pattern Recognition

**Multi-task Graph Attention Network for real-time football tactical analysis**  
Rishabh Karnawat · Gisma University of Applied Sciences · 2025

---

## What it does

TacticalGAT takes a 15-second window of a football match and simultaneously predicts three things:

| Head | Task | Output |
|------|------|--------|
| HEAD 1 | What tactic is the team currently playing? | `build_up` · `counter_attack` · `high_press` · `low_block` |
| HEAD 2 | Is the team adapting away from their usual style? | `adapting` · `not adapting` |
| HEAD 3 | What tactic should they switch to given the game state? | Same 4 classes |

Each match window is converted into a **player interaction graph** — 22 nodes (one per player), edges connecting players within 25 metres of each other, node features encoding positions and velocities. A shared Graph Attention Network encoder processes this graph; three independent heads read from the shared representation.

---

## Architecture

![TacticalGAT Architecture](docs/architecture_diagram.png)

```
Input graph (22 nodes × 5 features)
  └─ GAT Layer 1  [5 → 64,  8 heads, concat, ReLU, Dropout]
  └─ GAT Layer 2  [64 → 64, 8 heads, concat, ReLU, Dropout]
  └─ GAT Layer 3  [64 → 64, 8 heads, concat, ReLU, Dropout]
  └─ GAT Layer 4  [64 → 64, 8 heads, avg,    ReLU]
  └─ Global Mean Pool  →  graph vector (64 dims)
  └─ Concat with context (7 dims)  →  Fusion FC (71 → 128)
       ├─ HEAD 1: FC(128→64→4)   [tactic,     weight 1.0]
       ├─ HEAD 2: FC(128→32→2)   [adaptation, weight 0.5]
       └─ HEAD 3: FC(128→64→4)   [suggestion, weight 0.5]

Total parameters: ~86,000
```

**Context vector (7 dims):** score differential, match minute, home/away flag, team's historical tactic distribution (4 values).

**Architecture choices confirmed by ablation study** — see [`docs/ablation_study.docx`](docs/ablation_study.docx):
- ReLU over ELU: **+8.4 pp** tactic accuracy
- 4 GAT layers over 3: **+19.3 pp** in ablation
- 25 m proximity over 15 m: **+2.3 pp** tactic accuracy

---

## Data

**Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) — freely available, no licence required.

| Competition | Season | Matches |
|-------------|--------|---------|
| La Liga | 2015/16 | 380 |
| FIFA World Cup | 2022 | 64 |
| Copa América | 2024 | 31 |

- **475 matches total** → ~240,000 clips (15-second non-overlapping windows)
- Split **by match** (not by clip) to prevent data leakage: 70% train / 15% val / 15% test

**Tactic labels** are rule-based (no expert annotation needed):

| Tactic | Rule |
|--------|------|
| `high_press` | ≥ 2 pressures in opponent half |
| `counter_attack` | ≥ 1 ball recovery AND ≥ 1 progressive carry |
| `low_block` | ≤ 1 total pressures, ≥ 3 passes, ≥ 70% from own half |
| `build_up` | Default — none of the above |

---

## Class Imbalance Fix

Raw data is severely imbalanced (`build_up` = 71%, `high_press` = 0.02%).  
Two fixes are applied — both required for non-zero F1 on rare classes:

**1. Inverse-frequency class weights** on HEAD 1 cross-entropy loss:
```python
# high_press gets ~4000× the weight of build_up
class_weights = n_total / (n_classes × class_count)
loss_t = F.cross_entropy(out_t, y_t, weight=class_weights_tensor)
```

**2. Oversampling** rare classes in the training set with Gaussian noise augmentation on event-count features, targeting 3,000 clips per rare class.

---

## Results

### Main test results (after class imbalance fixes)

| Head | Task | Accuracy | Macro F1 |
|------|------|----------|----------|
| HEAD 1 | Tactic classifier (4-class) | — | — |
| HEAD 2 | Adaptation flag (binary) | — | — |
| HEAD 3 | Suggestion engine (4-class) | — | — |

> Run the notebook or `train.py` to populate these with your actual results.  
> Baseline (no fixes): HEAD 1 accuracy 90.6%, macro F1 **0.447** (counter_attack F1=0.00, high_press F1=0.00).

### Architecture comparison (5-epoch baseline)

| Model | Spatial | Attention | Params | Tac Acc | Tac F1 |
|-------|---------|-----------|--------|---------|--------|
| MLP | ✗ | ✗ | 19,466 | 71.9% | 0.242 |
| LSTM | ✗ | ✗ | 77,962 | 96.8%* | 0.481 |
| GCN | ✓ | ✗ | 39,146 | 91.0% | 0.445 |
| GraphSAGE | ✓ | ✗ | 47,658 | 92.3% | 0.454 |
| **GAT (ours)** | ✓ | ✓ | 69,098 | 90.6% | 0.447 |

*LSTM inflated by majority-class bias.

---

## Repo structure

```
TPR-GAT/
├── src/
│   ├── config.py          # All hyperparameters and constants
│   ├── data_loader.py     # StatsBomb download + clip generation
│   ├── labels.py          # Tactic labeling, adaptation flag, suggestion
│   ├── dataset.py         # Graph building, oversampling, DataLoaders
│   ├── model.py           # TacticalGAT + GATWithAttention
│   ├── train.py           # Training loop + full pipeline entry point
│   ├── evaluate.py        # Test evaluation, confusion matrices, attention viz
│   └── eda.py             # Exploratory data analysis plots
├── docs/
│   ├── architecture_diagram.png
│   └── ablation_study.docx
├── results/               # Generated figures (gitignored if large)
│   ├── confusion_matrices.png
│   ├── training_history.png
│   └── attention_pitch.png
├── rishabh_thesis1.ipynb  # Original notebook (full pipeline)
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

```
torch>=2.0
torch-geometric
statsbombpy
mplsoccer
scikit-learn
pandas
numpy
matplotlib
seaborn
networkx
```

### Run full pipeline

```bash
# From the TPR-GAT directory
python src/train.py
```

This runs everything end-to-end:  
download data → build clips → label → split → oversample → build graphs → train → evaluate → save figures.

### Run individual modules

```bash
# Download data only
python src/data_loader.py

# EDA plots (after downloading)
python src/eda.py

# Test label rules
python src/labels.py

# Check model architecture
python src/model.py
```

### Google Colab

Open `rishabh_thesis1.ipynb` directly in Colab. All dependencies install in the first cell.

---

## Key design decisions

**Why GAT over GCN?**  
GCN uses uniform aggregation — every neighbour contributes equally. GAT learns which player-to-player relationships matter most for each tactic. The attention weights can be visualised on a pitch diagram (see `results/attention_pitch.png`), giving interpretable per-prediction explanations. No other model in the comparison offers this.

**Why graph-based at all?**  
The MLP baseline (no graph structure) achieves 71.9% tactic accuracy — essentially the majority-class rate. Any graph model clears 90%. Spatial relationships between players are the signal; raw feature lists are not enough.

**Why rule-based labels?**  
No ground-truth tactic annotations exist in StatsBomb open data. Expert annotation at scale is not feasible. The rules are motivated by football domain knowledge, validated against known team tactical tendencies, and are fully reproducible — anyone can re-run the labeling.

---

## Limitations

- Labels are rule-based pseudo-ground-truth, not expert annotations
- StatsBomb open data has no tracking data — velocities are estimated from carry events only
- `build_up` dominates even after oversampling; rare-class F1 depends on threshold tuning
- Single training run — results vary with random seed
- Generalisability to other tactical styles (e.g. Bundesliga, Premier League) not verified

---

## Citation

```bibtex
@mastersthesis{karnawat2025tacticalgat,
  author = {Karnawat, Rishabh},
  title  = {TacticalGAT: Multi-Task Graph Attention Network for Football Tactical Pattern Recognition},
  school = {Gisma University of Applied Sciences},
  year   = {2025}
}
```

---

## Acknowledgements

- [StatsBomb](https://statsbomb.com) for the open event data
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) for the GAT implementation
- [mplsoccer](https://mplsoccer.readthedocs.io) for pitch visualisation
