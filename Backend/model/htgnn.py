"""
Architecture
------------

Mathematical formulation (see project report §4.3)
  m_{j→i}^(r) = Attn(h_j, h_i, e_{ji}^(r))          [heterogeneous msg]
  t'           = sin(ωt + φ)                           [time2vec encoding]
  h_i^(l+1)   = h_i^(l) + Σ_r AGG_r(m_{j→i}^(r))   [residual update]
  P(phishing)  = softmax(W · h_target)                [classification]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


# ── Temporal Attention ────────────────────────────────────────────────────────

    # edge_time_feat : (E, time_dim) – last 4 dims of edge_attr (time2vec output)
    # node_emb       : (N, hidden)


class TemporalAttention(nn.Module):
  

    def __init__(self, hidden_dim: int, time_dim: int = 4):
        super().__init__()
        self.W_time = nn.Linear(time_dim, hidden_dim, bias=False)
        self.gate   = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, node_emb: Tensor, time_emb: Tensor) -> Tensor:
        # time_emb: (N, hidden) – aggregated time signal per node
        t = self.W_time(time_emb)           # (N, hidden)
        gate = torch.sigmoid(self.gate(torch.cat([node_emb, t], dim=-1)))
        return gate * node_emb + (1 - gate) * t


# ── Main HT-GNN ───────────────────────────────────────────────────────────────
"""
    Parameters
    ----------
    metadata    : (node_types, edge_types) from HeteroData.metadata()
    eoa_in      : input feature dim for EOA nodes          (default 10)
    contract_in : input feature dim for Contract nodes     (default 8)
    hidden_dim  : internal embedding dimension             (default 64)
    num_heads   : attention heads in each HGTConv layer    (default 4)
    num_layers  : number of HGTConv layers                 (default 2)
    dropout     : dropout probability                      (default 0.3)
    """

class HTGNN(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        eoa_in: int = 10,
        contract_in: int = 8,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # ── input projections
        self.proj = nn.ModuleDict({
            "eoa":      Linear(eoa_in, hidden_dim),
            "contract": Linear(contract_in, hidden_dim),
        })
        # ── HGT layers ─
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])
        # ── temporal attention 
        self.temp_attn = nn.ModuleList([
            TemporalAttention(hidden_dim, time_dim=4)
            for _ in range(num_layers)
        ])
        # ── dropout & layer norms ────────────
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                "eoa":      nn.LayerNorm(hidden_dim),
                "contract": nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

        # ── classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),          # [benign, phishing]
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _aggregate_time_signal(
        self, data: HeteroData, node_type: str, num_nodes: int
    ) -> Tensor:
      
        device = data[node_type].x.device
        agg = torch.zeros(num_nodes, 4, device=device)
        count = torch.zeros(num_nodes, 1, device=device)

        for edge_type in data.edge_types:
            stype, _, dtype = edge_type
            if not hasattr(data[edge_type], "edge_attr"):
                continue
            edge_attr = data[edge_type].edge_attr   # (E, 6)
            if edge_attr.shape[1] < 6:
                continue
            time_feats = edge_attr[:, 2:]           # last 4 dims = time2vec
            edge_index = data[edge_type].edge_index

            if dtype == node_type:
                dst_idx = edge_index[1]
                agg.index_add_(0, dst_idx, time_feats)
                count.index_add_(0, dst_idx, torch.ones(dst_idx.size(0), 1, device=device))
            if stype == node_type:
                src_idx = edge_index[0]
                agg.index_add_(0, src_idx, time_feats)
                count.index_add_(0, src_idx, torch.ones(src_idx.size(0), 1, device=device))

        safe_count = count.clamp(min=1)
        return agg / safe_count

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, data: HeteroData) -> dict[str, Tensor]:
        # ── initial projection 
        x_dict: dict[str, Tensor] = {}
        for ntype, proj in self.proj.items():
            if hasattr(data[ntype], "x"):
                x_dict[ntype] = F.relu(proj(data[ntype].x))

        edge_index_dict = {et: data[et].edge_index
                           for et in data.edge_types
                           if hasattr(data[et], "edge_index")}
        # ── HGT message-passing layers 
        if edge_index_dict and any(edge.numel() > 0 for edge in edge_index_dict.values()):
            for l, (conv, temp, norms_l) in enumerate(
                zip(self.convs, self.temp_attn, self.norms)
            ):
                try:
                    out_dict = conv(x_dict, edge_index_dict)
                except RuntimeError:
                    # Sparse or tiny graphs can trip HGTConv internals.
                    # In that case, fall back to the current projected embeddings.
                    break

                # residual + temporal gating + layer norm
                new_x: dict[str, Tensor] = {}
                for ntype in x_dict:
                    if ntype not in out_dict or out_dict[ntype] is None:
                        new_x[ntype] = x_dict[ntype]
                        continue
                    h = out_dict[ntype]
                    # temporal attention
                    t_sig = self._aggregate_time_signal(data, ntype, h.size(0))
                    h = temp(h, t_sig)
                    # residual + norm
                    h = norms_l[ntype](h + x_dict[ntype])
                    h = self.dropout(h)
                    new_x[ntype] = h
                x_dict = new_x
        # ── extract target-node embedding ─────────────────────────────────────
        target_emb = None
        for ntype in ("eoa", "contract"):
            if not hasattr(data[ntype], "target_mask"):
                continue
            mask = data[ntype].target_mask
            if mask.any():
                target_emb = x_dict[ntype][mask]           # (1, hidden)
                break
        if target_emb is None:
            # Fallback: mean pool all EOA embeddings
            target_emb = x_dict["eoa"].mean(dim=0, keepdim=True)
        # ── classification ──────
        logits = self.classifier(target_emb)                # (1, 2)
        proba  = F.softmax(logits, dim=-1)
        return {
            "logits":     logits,
            "proba":      proba,
            "phishing_p": proba[0, 1].item(),
            "embeddings": x_dict,
        }


# ── Model loader ──────────────────────────────────────────────────────────────

_MODEL_INSTANCE: HTGNN | None = None

# Canonical metadata: all node types + edge types the model was trained with.
# Must match exactly what the training pipeline produced.
TRAIN_METADATA = (
    ["eoa", "contract"],
    [
        ("eoa",      "sends",    "eoa"),
        ("eoa",      "calls",    "contract"),
        ("contract", "returns",  "eoa"),
        ("eoa",      "approves", "contract"),
    ],
)


def load_model(path: str, device: str = "cpu") -> HTGNN:
    """
    Load the trained HT-GNN checkpoint.
    Falls back to an untrained model (all random weights) if the
    checkpoint file is not found – useful for development / testing.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    # The saved checkpoint was trained with a larger architecture than the
    # defaults used for ad-hoc development. Try the tuned configuration first
    # so the backend does not silently fall back to random weights.
    candidate_models = [
        HTGNN(metadata=TRAIN_METADATA, hidden_dim=128, num_heads=8, num_layers=2, dropout=0.3),
        HTGNN(metadata=TRAIN_METADATA),
    ]
    last_error: Exception | None = None

    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"[HT-GNN] WARNING – checkpoint not found at {path}. "
              "Using untrained model (heuristic fallback active).")
        model = candidate_models[-1]
    except Exception as e:
        print(f"[HT-GNN] WARNING – could not read checkpoint: {e}. "
              "Using untrained model.")
        model = candidate_models[-1]
    else:
        model = candidate_models[-1]
        for candidate in candidate_models:
            try:
                candidate.load_state_dict(state)
                model = candidate
                print(f"[HT-GNN] Loaded weights from {path} using {candidate.hidden_dim}-dim architecture")
                break
            except Exception as e:
                last_error = e
        else:
            print(f"[HT-GNN] WARNING – could not load weights: {last_error}. "
                  "Using untrained model.")

    model.eval()
    _MODEL_INSTANCE = model
    return model


def get_model() -> HTGNN:
    from config import get_settings
    cfg = get_settings()
    return load_model(cfg.model_path, cfg.model_device)
