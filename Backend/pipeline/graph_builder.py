"""
chainguard/backend/pipeline/graph_builder.py

Converts raw Ethereum data (from fetcher.py) into a PyTorch Geometric
HeteroData object ready for the HT-GNN model.

Node types : "eoa"      – Externally Owned Account
             "contract" – Smart contract

Edge types  : ("eoa",      "sends",    "eoa")
              ("eoa",      "calls",    "contract")
              ("contract", "returns",  "eoa")
              ("eoa",      "approves", "contract")   ← drainer risk signal
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from config import get_settings

cfg = get_settings()

# ── feature dimensions ────────────────────────────────────────────────────────
EOA_FEAT_DIM = 10
CONTRACT_FEAT_DIM = 8
EDGE_FEAT_DIM = 6       # timestamp-encoded + value + type flags


# ── time2vec temporal encoding ────────────────────────────────────────────────

def time2vec(timestamp: int, dim: int = 4) -> list[float]:
    """
    Sinusoidal temporal encoding (Time2Vec-lite).
    Maps a Unix timestamp to `dim` floats in [-1, 1].
    """
    t = float(timestamp) / 1e9          # normalise to ~[0, ~50]
    out = [t]                           # linear component
    for k in range(1, dim):
        freq = 2 ** k
        out.append(math.sin(freq * t))
    return out                          # length = dim


# ── per-node feature extraction ───────────────────────────────────────────────

def _eoa_features(node_data: dict, all_txs: list[dict]) -> list[float]:
    
    balance = node_data.get("balance_eth", 0.0)
    txs = all_txs or node_data.get("txs", [])
    nonce = node_data.get("nonce", len(txs))
    first_block = node_data.get("first_block", 0)
    inflow = [t for t in txs if t.get("to", "").lower() == node_data.get("address", "")]
    outflow = [t for t in txs if t.get("from", "").lower() == node_data.get("address", "")]
    counterparts = {t.get("from", "") for t in txs} | {t.get("to", "") for t in txs}
    values = [int(t.get("value", 0)) / 1e18 for t in txs if int(t.get("value", 0)) > 0]
    avg_val = sum(values) / len(values) if values else 0.0
    has_token = float(bool(node_data.get("token_txs")))
    age_norm = min(first_block / 20_000_000, 1.0)   # normalise to ~[0,1]
    return [
        math.log1p(balance),
        math.log1p(len(txs)),
        float(len(inflow)),
        float(len(outflow)),
        len(inflow) / max(len(txs), 1),
        math.log1p(avg_val),
        age_norm,
        math.log1p(len(counterparts)),
        has_token,
        math.log1p(nonce),
    ]
def _contract_features(node_data: dict) -> list[float]:
    txs = node_data.get("txs", [])
    approvals = node_data.get("approvals", [])
    addr = node_data.get("address", "")
    inflow = [t for t in txs if (t.get("to") or "").lower() == addr]
    outflow = [t for t in txs if (t.get("from") or "").lower() == addr]
    callers = {t.get("from", "") for t in txs}
    values = [int(t.get("value", 0)) / 1e18 for t in txs if int(t.get("value", 0)) > 0]
    avg_val = sum(values) / len(values) if values else 0.0
    has_internal = float(bool(node_data.get("internal_txs")))

    return [
        math.log1p(len(txs)),
        float(len(inflow)),
        float(len(outflow)),
        len(inflow) / max(len(txs), 1),
        math.log1p(avg_val),
        math.log1p(len(approvals)),
        math.log1p(len(callers)),
        has_internal,
    ]
def _edge_features(tx: dict) -> list[float]:
    value = math.log1p(int(tx.get("value", 0)) / 1e18)
    gas = math.log1p(int(tx.get("gasPrice", tx.get("gas_price", 1))) / 1e9)
    ts = int(tx.get("timeStamp", tx.get("timestamp", int(time.time()))))
    tvec = time2vec(ts, dim=4)
    return [value, gas] + tvec


# ── graph builder ─────────────────────────────────────────────────────────────

class HeteroGraphBuilder:
    def build(self, raw: dict) -> tuple[HeteroData, dict]:
        target_info = raw["target"]
        hop1 = raw.get("hop1", [])
        hop2 = raw.get("hop2", [])
        # ── node registry ────────
        # Maps address → (node_type, local_index)
        node_registry: dict[str, tuple[str, int]] = {}
        eoa_feats: list[list[float]] = []
        contract_feats: list[list[float]] = []
        eoa_addrs: list[str] = []
        contract_addrs: list[str] = []
        def register_node(node_data: dict) -> tuple[str, int]:
            addr = node_data["address"].lower()
            if addr in node_registry:
                return node_registry[addr]
            if node_data.get("is_contract", False):
                ntype = "contract"
                idx = len(contract_addrs)
                contract_addrs.append(addr)
                contract_feats.append(_contract_features(node_data))
            else:
                ntype = "eoa"
                idx = len(eoa_addrs)
                eoa_addrs.append(addr)
                all_txs = (node_data.get("normal_txs") or
                           node_data.get("txs") or [])
                eoa_feats.append(_eoa_features(node_data, all_txs))
            node_registry[addr] = (ntype, idx)
            return ntype, idx
        # ── register target node ─────────────────────────────────────────────
        target_info["address"] = target_info["address"].lower()
        target_ntype, target_idx = register_node(target_info)

        # ── register hop-1 neighbours ────────────────────────────────────────
        for n in hop1:
            register_node(n)

        # ── register hop-2 neighbours ────────────────────────────────────────
        for n in hop2:
            addr = n["address"].lower()
            if addr not in node_registry:
                # Minimal info for hop-2 nodes
                register_node({"address": addr, "is_contract": False,
                               "balance_eth": 0, "txs": []})

        # ── edge lists ───────────────────────────────────────────────────────
        # Structure: edges[(src_type, rel, dst_type)] = ([src_idx], [dst_idx], [[feats]])
        edges: dict[tuple, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))

        def add_edge(tx: dict, from_addr: str, to_addr: str):
            from_addr, to_addr = from_addr.lower(), to_addr.lower()
            if from_addr not in node_registry or to_addr not in node_registry:
                return
            stype, sidx = node_registry[from_addr]
            dtype, didx = node_registry[to_addr]

            if stype == "eoa" and dtype == "eoa":
                rel = "sends"
            elif stype == "eoa" and dtype == "contract":
                rel = "calls"
            elif stype == "contract" and dtype == "eoa":
                rel = "returns"
            else:
                # The HT-GNN metadata only supports the three relations above.
                # Skip unsupported contract->contract edges instead of creating
                # an out-of-vocabulary relation that breaks HGTConv.
                return
            key = (stype, rel, dtype)
            edges[key][0].append(sidx)
            edges[key][1].append(didx)
            edges[key][2].append(_edge_features(tx))
        # Build edges from target's transactions
        target_addr = target_info["address"]
        for tx in target_info.get("normal_txs", []):
            add_edge(tx, tx.get("from", ""), tx.get("to", ""))
        # Approval edges (strong drainer signal)
        for app in target_info.get("approvals", []):
            spender = app.get("topics", [None, None, None])[2] or ""
            if len(spender) > 26:
                spender_addr = "0x" + spender[-40:]
                if spender_addr.lower() not in node_registry:
                    register_node({"address": spender_addr, "is_contract": True,
                                   "balance_eth": 0, "txs": []})
                fake_tx = {"value": "0", "gasPrice": "1", "timeStamp": str(int(time.time()))}
                key = ("eoa", "approves", "contract")
                t_type, t_idx = node_registry[target_addr]
                s_type, s_idx = node_registry.get(spender_addr.lower(), ("contract", 0))
                edges[key][0].append(t_idx)
                edges[key][1].append(s_idx)
                edges[key][2].append(_edge_features(fake_tx))

        # Build edges from hop-1 transactions
        for n in hop1:
            for tx in n.get("txs", []):
                add_edge(tx, tx.get("from", ""), tx.get("to", ""))

        # ── assemble HeteroData ─────────
        data = HeteroData()

        # Node features
        if eoa_feats:
            data["eoa"].x = torch.tensor(eoa_feats, dtype=torch.float)
        else:
            data["eoa"].x = torch.zeros((1, EOA_FEAT_DIM), dtype=torch.float)
            eoa_addrs = ["0x0000000000000000000000000000000000000000"]

        if contract_feats:
            data["contract"].x = torch.tensor(contract_feats, dtype=torch.float)
        else:
            data["contract"].x = torch.zeros((1, CONTRACT_FEAT_DIM), dtype=torch.float)
            contract_addrs = ["0x0000000000000000000000000000000000000000"]

        # Edge index + edge attr
        for (stype, rel, dtype), (srcs, dsts, feats) in edges.items():
            key = (stype, rel, dtype)
            if srcs:
                data[key].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
                data[key].edge_attr = torch.tensor(feats, dtype=torch.float)

        # Target node metadata
        data["eoa"].target_mask = torch.zeros(len(eoa_addrs), dtype=torch.bool)
        data["contract"].target_mask = torch.zeros(len(contract_addrs), dtype=torch.bool)
        if target_ntype == "eoa":
            data["eoa"].target_mask[target_idx] = True
        else:
            data["contract"].target_mask[target_idx] = True

        meta = {
            "target_type": target_ntype,
            "target_idx": target_idx,
            "num_eoa": len(eoa_addrs),
            "num_contracts": len(contract_addrs),
            "num_edges": sum(len(v[0]) for v in edges.values()),
            "eoa_addresses": eoa_addrs,
            "contract_addresses": contract_addrs,
            "edge_types": list(edges.keys()),
        }
        return data, meta
    """
    Converts a raw subgraph dict (from EthereumFetcher) into a
    PyG HeteroData for inference by the HT-GNN model.
    """