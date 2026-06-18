"""
model.py — TacticalGAT and all baseline/ablation model variants.

Classes
-------
TacticalGAT       — main model (best architecture from ablation)
GATWithAttention  — wraps TacticalGAT to expose layer-4 attention weights
MLPBaseline       — flat MLP, no graph structure
LSTMBaseline      — sequential LSTM over player nodes
GCNBaseline       — GCN with uniform aggregation
GraphSAGEVariant  — inductive GraphSAGE
AblationGAT       — parameterised variant for ablation experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, GCNConv, SAGEConv, global_mean_pool,
)

from config import (
    NODE_FEATURES, CONTEXT_FEATURES,
    HIDDEN_DIM, HEADS, N_CLASSES, DROPOUT, DEVICE,
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TacticalGAT(nn.Module):
    """
    Multi-task Graph Attention Network — best architecture from ablation.

    Encoder : 4 × GATConv layers, 8 heads each, ReLU, dropout 0.3
    Fusion  : cat(graph_64, context_7) -> Linear(71->128) -> ReLU -> Dropout
    HEAD 1  : tactic_label     4-class   loss weight 1.0
    HEAD 2  : adaptation_flag  binary    loss weight 0.5
    HEAD 3  : suggestion_label 4-class   loss weight 0.5

    Ablation findings applied:
      ReLU  over ELU  (+8.4 pp tactic accuracy)
      4 layers over 3 (+19.3 pp in ablation run)
      25 m proximity  (+2.3 pp tactic accuracy)
    """

    def __init__(self,
                 node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM,
                 heads=HEADS,
                 n_classes=N_CLASSES,
                 dropout=DROPOUT):
        super().__init__()
        self.dropout_rate     = dropout
        self.context_features = context_features
        head_dim = hidden_dim // heads   # 64 // 8 = 8

        self.gat1 = GATConv(node_features, head_dim,
                            heads=heads, concat=True,  dropout=dropout)
        self.gat2 = GATConv(hidden_dim,   head_dim,
                            heads=heads, concat=True,  dropout=dropout)
        self.gat3 = GATConv(hidden_dim,   head_dim,
                            heads=heads, concat=True,  dropout=dropout)
        self.gat4 = GATConv(hidden_dim,   hidden_dim,
                            heads=heads, concat=False, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_features, 128),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_tactic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.head_adapt = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self.head_suggest = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        context       = data.context
        bs            = batch.max().item() + 1

        if context.dim() == 1 and context.numel() == bs * self.context_features:
            context = context.view(bs, self.context_features)
        elif context.dim() == 3:
            context = context.squeeze(1)

        x = F.relu(self.gat1(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.gat2(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.gat3(x, ei))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.gat4(x, ei))

        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, context], dim=1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GATWithAttention(nn.Module):
    """Wraps TacticalGAT to also return layer-4 attention weights for visualisation."""

    def __init__(self, base: TacticalGAT):
        super().__init__()
        self.gat1 = base.gat1; self.gat2 = base.gat2
        self.gat3 = base.gat3; self.gat4 = base.gat4
        self.fusion       = base.fusion
        self.head_tactic  = base.head_tactic
        self.head_adapt   = base.head_adapt
        self.head_suggest = base.head_suggest
        self.dr  = base.dropout_rate
        self.ctx = base.context_features

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        context       = data.context
        bs = batch.max().item() + 1

        if context.dim() == 1 and context.numel() == bs * self.ctx:
            context = context.view(bs, self.ctx)
        elif context.dim() == 3:
            context = context.squeeze(1)

        x = F.relu(self.gat1(x, ei))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = F.relu(self.gat2(x, ei))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = F.relu(self.gat3(x, ei))
        x = F.dropout(x, p=self.dr, training=self.training)
        x, (att_ei, att_w) = self.gat4(x, ei, return_attention_weights=True)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, context], dim=1))
        return (self.head_tactic(x), self.head_adapt(x),
                self.head_suggest(x), att_ei, att_w)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE MODELS (architecture comparison, Experiment 1)
# ══════════════════════════════════════════════════════════════════════════════

class _BaselineHeads(nn.Module):
    """Shared three-head readout used by all baselines."""
    def __init__(self, in_dim, n_classes=N_CLASSES):
        super().__init__()
        self.head_tactic  = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                           nn.Linear(64, n_classes))
        self.head_adapt   = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(),
                                           nn.Linear(32, 2))
        self.head_suggest = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                           nn.Linear(64, n_classes))


class MLPBaseline(nn.Module):
    """Flat MLP — no graph structure. Lower bound for comparison."""
    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES, n_classes=N_CLASSES):
        super().__init__()
        in_dim = node_features + context_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
        )
        self.heads = _BaselineHeads(128, n_classes)
        self.ctx   = context_features

    def forward(self, data):
        bs      = data.batch.max().item() + 1
        context = data.context
        if context.dim() == 1:
            context = context.view(bs, self.ctx)
        elif context.dim() == 3:
            context = context.squeeze(1)

        # Mean pool node features without graph structure
        node_means = global_mean_pool(data.x, data.batch)
        x = self.mlp(torch.cat([node_means, context], dim=1))
        return self.heads.head_tactic(x), self.heads.head_adapt(x), self.heads.head_suggest(x)


class LSTMBaseline(nn.Module):
    """LSTM over player nodes (treats graph as a sequence)."""
    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(node_features, hidden_dim, num_layers=2,
                            batch_first=True)
        self.fusion = nn.Linear(hidden_dim + context_features, 128)
        self.heads  = _BaselineHeads(128, n_classes)
        self.ctx    = context_features

    def forward(self, data):
        bs      = data.batch.max().item() + 1
        context = data.context
        if context.dim() == 1:
            context = context.view(bs, self.ctx)
        elif context.dim() == 3:
            context = context.squeeze(1)

        # Reshape node features into (batch, 22, node_features)
        x = data.x.view(bs, -1, data.x.size(1))
        _, (h, _) = self.lstm(x)
        h = h[-1]  # last layer hidden state
        x = F.relu(self.fusion(torch.cat([h, context], dim=1)))
        return self.heads.head_tactic(x), self.heads.head_adapt(x), self.heads.head_suggest(x)


class GCNBaseline(nn.Module):
    """GCN — graph structure but uniform aggregation (no attention)."""
    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.conv1   = GCNConv(node_features, hidden_dim)
        self.conv2   = GCNConv(hidden_dim,    hidden_dim)
        self.fusion  = nn.Sequential(nn.Linear(hidden_dim + context_features, 128), nn.ReLU())
        self.heads   = _BaselineHeads(128, n_classes)
        self.ctx     = context_features

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        context       = data.context
        bs = batch.max().item() + 1
        if context.dim() == 1:
            context = context.view(bs, self.ctx)
        elif context.dim() == 3:
            context = context.squeeze(1)

        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, context], dim=1))
        return self.heads.head_tactic(x), self.heads.head_adapt(x), self.heads.head_suggest(x)


class GraphSAGEVariant(nn.Module):
    """GraphSAGE — inductive, samples neighbours during aggregation."""
    def __init__(self, node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.conv1  = SAGEConv(node_features, hidden_dim)
        self.conv2  = SAGEConv(hidden_dim,    hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + context_features, 128), nn.ReLU())
        self.heads  = _BaselineHeads(128, n_classes)
        self.ctx    = context_features

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        context       = data.context
        bs = batch.max().item() + 1
        if context.dim() == 1:
            context = context.view(bs, self.ctx)
        elif context.dim() == 3:
            context = context.squeeze(1)

        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, context], dim=1))
        return self.heads.head_tactic(x), self.heads.head_adapt(x), self.heads.head_suggest(x)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION VARIANT (Experiment 3)
# ══════════════════════════════════════════════════════════════════════════════

class AblationGAT(nn.Module):
    """
    Parameterised TacticalGAT for ablation study.
    Vary: n_layers, activation ('relu'|'elu'|'leaky_relu'|'tanh')
    Proximity threshold is set at graph-build time (not inside model).
    """

    def __init__(self,
                 node_features=NODE_FEATURES,
                 context_features=CONTEXT_FEATURES,
                 hidden_dim=HIDDEN_DIM,
                 heads=HEADS,
                 n_classes=N_CLASSES,
                 dropout=DROPOUT,
                 n_layers=4,
                 activation="relu"):
        super().__init__()
        self.dropout_rate     = dropout
        self.context_features = context_features
        self.activation       = activation
        head_dim = hidden_dim // heads

        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim  = node_features if i == 0 else hidden_dim
            is_last = (i == n_layers - 1)
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim if is_last else head_dim,
                        heads=1 if is_last else heads,
                        concat=not is_last,
                        dropout=dropout)
            )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_features, 128),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_tactic  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
        self.head_adapt   = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))
        self.head_suggest = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def _act(self, x):
        if self.activation == "relu":       return F.relu(x)
        if self.activation == "elu":        return F.elu(x)
        if self.activation == "leaky_relu": return F.leaky_relu(x, 0.2)
        if self.activation == "tanh":       return torch.tanh(x)
        return F.relu(x)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        context       = data.context
        bs = batch.max().item() + 1

        if context.dim() == 1 and context.numel() == bs * self.context_features:
            context = context.view(bs, self.context_features)
        elif context.dim() == 3:
            context = context.squeeze(1)

        for i, layer in enumerate(self.gat_layers):
            x = self._act(layer(x, ei))
            if i < len(self.gat_layers) - 1:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, context], dim=1))
        return self.head_tactic(x), self.head_adapt(x), self.head_suggest(x)


if __name__ == "__main__":
    model = TacticalGAT().to(DEVICE)
    print(f"TacticalGAT ready on {DEVICE}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Architecture: 4 GAT layers | ReLU | 25 m proximity | 8 heads | dim=64")
