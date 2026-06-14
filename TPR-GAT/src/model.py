"""
model.py
All model definitions:
  - TacticalGAT      : primary three-head GAT (this thesis)
  - MLPBaseline
  - LSTMBaseline
  - GCNBaseline
  - GraphSAGEVariant
  - TacticalGAT_CustomActivation  (used in ablation study)
  - TacticalGAT_NLayers           (used in ablation study)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

from src.config import (
    NODE_FEATURES, CONTEXT_FEATURES, HIDDEN_DIM, GAT_HEADS,
    N_CLASSES, DROPOUT
)


# ── Shared context-handling helper ─────────────────────────────────────────────

def _fix_context(context: torch.Tensor, batch: torch.Tensor,
                 n_features: int) -> torch.Tensor:
    """
    PyG batches the context tensor in various shapes depending on version.
    This normalises it to (batch_size, n_features) every time.
    """
    batch_size = batch.max().item() + 1
    if context.dim() == 1 and context.numel() == batch_size * n_features:
        return context.view(batch_size, n_features)
    if context.dim() == 3:
        return context.squeeze(1)
    return context


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TacticalGAT(nn.Module):
    """
    Multi-task Graph Attention Network — the main thesis model.

    Architecture
    ────────────
    Shared encoder  : 3 x GATConv (8 heads each)
    Graph pooling   : global mean pool  →  (batch, 64)
    Context fusion  : cat(graph_vec, context_vec)  →  Linear → (batch, 128)
    HEAD 1          : tactic_label      4-class  (primary)
    HEAD 2          : adaptation_flag   2-class  (binary)
    HEAD 3          : suggestion_label  4-class

    Parameters
    ──────────
    node_features    : 5  (x, y, vx, vy, team_flag)
    context_features : 7  (score, minute, home, 4x tactic history)
    hidden_dim       : 64
    heads            : 8
    n_classes        : 4
    dropout          : 0.3
    """

    def __init__(self,
                 node_features:    int   = NODE_FEATURES,
                 context_features: int   = CONTEXT_FEATURES,
                 hidden_dim:       int   = HIDDEN_DIM,
                 heads:            int   = GAT_HEADS,
                 n_classes:        int   = N_CLASSES,
                 dropout:          float = DROPOUT):
        super().__init__()
        self.dropout_rate     = dropout
        self.context_features = context_features

        # GAT layer 1: 5 → 64  (8 heads × 8, concat=True)
        self.gat1 = GATConv(node_features,    hidden_dim // heads,
                            heads=heads, concat=True,  dropout=dropout)
        # GAT layer 2: 64 → 64
        self.gat2 = GATConv(hidden_dim,       hidden_dim // heads,
                            heads=heads, concat=True,  dropout=dropout)
        # GAT layer 3: 64 → 64  (concat=False averages the heads)
        self.gat3 = GATConv(hidden_dim,       hidden_dim,
                            heads=heads, concat=False, dropout=dropout)

        # Context fusion: (64 + 7) → 128
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_features, 128),
            nn.ReLU(), nn.Dropout(dropout)
        )

        # HEAD 1 — tactic classifier
        self.head_tactic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        # HEAD 2 — adaptation detector (binary)
        self.head_adapt = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        # HEAD 3 — suggestion engine
        self.head_suggest = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ctx = _fix_context(data.context, batch, self.context_features)

        x = F.elu(self.gat1(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.gat2(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.gat3(x, ei))

        x = global_mean_pool(x, batch)           # (batch, 64)
        x = self.fusion(torch.cat([x, ctx], 1))  # (batch, 128)

        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES (used in architecture comparison)
# ══════════════════════════════════════════════════════════════════════════════

class MLPBaseline(nn.Module):
    """Flat MLP — no graph structure. Lower-bound baseline."""

    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden=128, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.context_features = context_features
        self.shared = nn.Sequential(
            nn.Linear(node_features + context_features, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_tactic  = nn.Linear(hidden, n_classes)
        self.head_adapt   = nn.Linear(hidden, 2)
        self.head_suggest = nn.Linear(hidden, n_classes)

    def forward(self, data):
        x   = global_mean_pool(data.x, data.batch)
        ctx = _fix_context(data.context, data.batch, self.context_features)
        x   = self.shared(torch.cat([x, ctx], 1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


class LSTMBaseline(nn.Module):
    """LSTM over node sequence. Temporal ordering, no spatial edges."""

    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden=64, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.context_features = context_features
        self.hidden = hidden
        self.lstm = nn.LSTM(node_features, hidden, batch_first=True,
                            dropout=dropout, num_layers=2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + context_features, 128),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.head_tactic  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))
        self.head_adapt   = nn.Linear(128, 2)
        self.head_suggest = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, data):
        bs = data.batch.max().item() + 1
        max_n, graph_nodes = 0, []
        for g in range(bs):
            nodes = data.x[data.batch == g]
            graph_nodes.append(nodes)
            max_n = max(max_n, nodes.shape[0])
        padded = torch.zeros(bs, max_n, data.x.shape[1], device=data.x.device)
        for i, nodes in enumerate(graph_nodes):
            padded[i, :nodes.shape[0]] = nodes
        _, (hn, _) = self.lstm(padded)
        x   = hn[-1]
        ctx = _fix_context(data.context, data.batch, self.context_features)
        x   = self.fusion(torch.cat([x, ctx], 1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


class GCNBaseline(nn.Module):
    """Graph Convolutional Network — same graph, uniform aggregation."""

    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.dropout_rate     = dropout
        self.context_features = context_features
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim,    hidden_dim)
        self.conv3 = GCNConv(hidden_dim,    hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_features, 128),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.head_tactic  = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))
        self.head_adapt   = nn.Sequential(
            nn.Linear(128,32), nn.ReLU(), nn.Linear(32,2))
        self.head_suggest = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ctx = _fix_context(data.context, batch, self.context_features)
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv3(x, ei))
        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, ctx], 1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


class GraphSAGEVariant(nn.Module):
    """GraphSAGE — inductive, no attention."""

    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.dropout_rate     = dropout
        self.context_features = context_features
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim,    hidden_dim)
        self.conv3 = SAGEConv(hidden_dim,    hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_features, 128),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.head_tactic  = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))
        self.head_adapt   = nn.Sequential(
            nn.Linear(128,32), nn.ReLU(), nn.Linear(32,2))
        self.head_suggest = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ctx = _fix_context(data.context, batch, self.context_features)
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv3(x, ei))
        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, ctx], 1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class TacticalGAT_CustomActivation(nn.Module):
    """GAT with a swappable activation function for ablation study."""

    def __init__(self, node_features=NODE_FEATURES, context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM, heads=GAT_HEADS, n_classes=N_CLASSES,
                 dropout=DROPOUT, activation_fn=F.elu):
        super().__init__()
        self.dropout_rate     = dropout
        self.context_features = context_features
        self.act              = activation_fn
        self.gat1 = GATConv(node_features,    hidden_dim//heads,
                            heads=heads, concat=True,  dropout=dropout)
        self.gat2 = GATConv(hidden_dim,       hidden_dim//heads,
                            heads=heads, concat=True,  dropout=dropout)
        self.gat3 = GATConv(hidden_dim,       hidden_dim,
                            heads=heads, concat=False, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_features, 128), nn.ReLU(), nn.Dropout(dropout))
        self.head_tactic  = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))
        self.head_adapt   = nn.Sequential(
            nn.Linear(128,32), nn.ReLU(), nn.Linear(32,2))
        self.head_suggest = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ctx = _fix_context(data.context, batch, self.context_features)
        x = self.act(self.gat1(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.act(self.gat2(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.act(self.gat3(x, ei))
        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, ctx], 1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


class TacticalGAT_NLayers(nn.Module):
    """GAT with configurable number of layers for ablation study."""

    def __init__(self, node_features=NODE_FEATURES, context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM, heads=GAT_HEADS, n_classes=N_CLASSES,
                 dropout=DROPOUT, n_layers=3):
        super().__init__()
        self.dropout_rate     = dropout
        self.context_features = context_features
        self.convs            = nn.ModuleList()

        for i in range(n_layers):
            in_ch = node_features if i == 0 else hidden_dim
            if i < n_layers - 1:
                self.convs.append(GATConv(in_ch, hidden_dim//heads,
                                          heads=heads, concat=True, dropout=dropout))
            else:
                self.convs.append(GATConv(in_ch, hidden_dim,
                                          heads=heads, concat=False, dropout=dropout))

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_features, 128), nn.ReLU(), nn.Dropout(dropout))
        self.head_tactic  = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))
        self.head_adapt   = nn.Sequential(
            nn.Linear(128,32), nn.ReLU(), nn.Linear(32,2))
        self.head_suggest = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,n_classes))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ctx = _fix_context(data.context, batch, self.context_features)
        for i, conv in enumerate(self.convs):
            x = F.elu(conv(x, ei))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, ctx], 1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


if __name__ == "__main__":
    model = TacticalGAT()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TacticalGAT parameters: {n:,}")
