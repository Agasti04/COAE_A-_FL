import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# General Notes:
# - SharedTower: only uses shared features (global)
# - LocalTower: uses local features + detached shared representation
# - Combiner: merges shared + local for final 3-class output
# - All layers include BatchNorm + Dropout for stability
# ---------------------------------------------------------

# common dropout probability
DROPOUT_P = 0.3


# ---------------------------------------------------------
# Shared Tower
# ---------------------------------------------------------
class SharedTower(nn.Module):
    def __init__(self, in_features: int, embed_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Local Tower
# ---------------------------------------------------------
class LocalTower(nn.Module):
    def __init__(self, in_features: int, shared_dim: int = 64, out_dim: int = 64):
        super().__init__()

        # Local features first
        self.local_layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),
        )

        # Combine with shared representation
        combined_in = 128 + shared_dim

        self.combined_layers = nn.Sequential(
            nn.Linear(combined_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, local_x, shared_rep_detached):
        lx = self.local_layers(local_x)
        combined = torch.cat([lx, shared_rep_detached], dim=1)
        return self.combined_layers(combined)


# ---------------------------------------------------------
# Combiner
# combines shared representation + local representation
# final output = 3-class logits
# ---------------------------------------------------------
class Combiner(nn.Module):
    def __init__(self, shared_dim: int = 64, local_dim: int = 64, num_classes: int = 3):
        super().__init__()

        in_dim = shared_dim + local_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),

            nn.Linear(64, num_classes)  # final logits (no softmax)
        )

    def forward(self, shared_rep, local_rep):
        x = torch.cat([shared_rep, local_rep], dim=1)
        return self.net(x)
