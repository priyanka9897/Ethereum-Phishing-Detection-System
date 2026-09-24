"""
chainguard/backend/model/train.py

Training pipeline for the HT-GNN phishing detector.

Usage
-----
  python train.py --dataset ./data/phishing_dataset.csv \
                  --epochs 100 \
                  --output ./model/htgnn_phishing.pt

Dataset format (CSV)
--------------------
  address,label       (label: 1 = phishing, 0 = benign)

The script fetches on-chain data for each address (requires API keys
in .env), builds HeteroData graphs, and trains with:
  - Binary cross-entropy loss
  - Contrastive regularisation (pull phishing apart from benign)
  - AdamW optimiser + cosine LR schedule
  - 80/10/10 train/val/test split stratified by label

For large datasets (>10k addresses) consider pre-fetching and caching
the subgraph raw dicts to disk before training.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings
from pipeline.fetcher import EthereumFetcher
from pipeline.graph_builder import HeteroGraphBuilder
from model.htgnn import HTGNN, TRAIN_METADATA

log = logging.getLogger(__name__)
cfg = get_settings()


# ── contrastive loss helper ───────────────────────────────────────────────────

def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
    """
    Pairwise contrastive loss encouraging phishing embeddings to cluster
    away from benign ones.
    """
    n = embeddings.size(0)
    if n < 2:
        return torch.tensor(0.0)
    # Euclidean distance matrix
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)   # (n,n,d)
    dist = diff.pow(2).sum(-1).sqrt()                          # (n,n)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    loss  = same * dist.pow(2)
    loss += (1 - same) * F.relu(margin - dist).pow(2)
    return loss.mean()


# ── dataset helpers ───────────────────────────────────────────────────────────

def load_csv(path: str) -> list[tuple[str, int]]:
    """
    Load an address/label CSV.

    Supports either the canonical lowercase columns used by the trainer
    (``address``, ``label``) or the benchmark-style columns found in the
    provided dataset (``Address``, ``FLAG``).
    """
    rows: list[tuple[str, int]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = (
                row.get("address")
                or row.get("Address")
                or row.get("ADDRESS")
                or ""
            ).strip().lower()
            label_raw = row.get("label") or row.get("Label") or row.get("FLAG") or row.get("flag")
            if not address or label_raw is None or label_raw == "":
                continue
            rows.append((address, int(float(label_raw))))
    return rows


async def fetch_and_build(addresses: list[tuple[str, int]]) -> list[tuple]:
    """
    Returns list of (HeteroData, label, address) tuples.
    Skips addresses where fetching fails.
    """
    builder = HeteroGraphBuilder()
    results = []
    async with EthereumFetcher() as fetcher:
        for addr, label in addresses:
            try:
                raw = await fetcher.collect_full_subgraph_data(addr)
                data, meta = builder.build(raw)
                results.append((data, label, addr))
                log.info("Built graph for %s (label=%d)", addr, label)
            except Exception as e:
                log.warning("Skipping %s: %s", addr, e)
    return results


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    phishing = [d for d in dataset if d[1] == 1]
    benign   = [d for d in dataset if d[1] == 0]
    random.shuffle(phishing); random.shuffle(benign)

    def _split(lst):
        n = len(lst)
        t = int(n * train_ratio)
        v = int(n * (train_ratio + val_ratio))
        return lst[:t], lst[t:v], lst[v:]

    tp, vp, ep = _split(phishing)
    tb, vb, eb = _split(benign)
    return tp+tb, vp+vb, ep+eb


# ── training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, dataset, optimizer, device):
    model.train()
    random.shuffle(dataset)
    total_loss = 0.0

    for data, label, _ in dataset:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        logits = out["logits"]                          # (1, 2)
        embeddings = out["embeddings"]["eoa"][0:1]      # (1, hidden)
        target = torch.tensor([label], device=device)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, target)
        
        # Contrastive loss to encourage better embedding separation
        contra_loss = contrastive_loss(embeddings, target, margin=1.0)
        
        # Combine losses: 80% CE + 20% contrastive for better generalization
        total_task_loss = 0.8 * ce_loss + 0.2 * contra_loss
        
        total_task_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += total_task_loss.item()

    return total_loss / max(len(dataset), 1)


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    probs, labels = [], []
    for data, label, _ in dataset:
        data = data.to(device)
        out = model(data)
        probs.append(out["phishing_p"])
        labels.append(label)

    preds = [1 if p >= 0.5 else 0 for p in probs]
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
    f1  = f1_score(labels, preds, zero_division=0)
    return auc, f1, probs, preds


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HT-GNN phishing detector")
    parser.add_argument("--dataset", required=True, help="CSV with address,label columns")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default=cfg.model_device)
    parser.add_argument("--output", default=cfg.model_path)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional maximum number of rows to use from the dataset")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Load pre-fetched graphs from ./data/graphs/*.pt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device)

    # ── load / fetch dataset ──────────────────────────────────────────────────
    if args.skip_fetch:
        graphs_dir = Path("./data/graphs")
        requested = load_csv(args.dataset)
        if args.limit and args.limit > 0:
            requested = requested[:args.limit]

        requested_addrs = [addr for addr, _ in requested]
        requested_set = set(requested_addrs)

        cache_by_addr = {}
        for pt_file in sorted(graphs_dir.glob("*.pt")):
            item = torch.load(pt_file, weights_only=False)
            addr = str(item.get("address", "")).strip().lower()
            if addr:
                cache_by_addr[addr] = item

        dataset = []
        missing = 0
        for addr, label in requested:
            item = cache_by_addr.get(addr)
            if item is None:
                missing += 1
                continue
            # Use dataset label to keep training target consistent with current CSV.
            dataset.append((item["data"], label, addr))

        log.info(
            "Offline mode: requested=%d cached_used=%d missing=%d",
            len(requested_addrs), len(dataset), missing
        )
        if not dataset:
            raise RuntimeError(
                "--skip-fetch requested but no matching cached graphs found in ./data/graphs"
            )
    else:
        addresses = load_csv(args.dataset)
        if args.limit and args.limit > 0:
            addresses = addresses[:args.limit]
        log.info("Fetching on-chain data for %d addresses …", len(addresses))
        dataset = asyncio.run(fetch_and_build(addresses))
        # Cache graphs
        cache_dir = Path("./data/graphs"); cache_dir.mkdir(parents=True, exist_ok=True)
        for data, label, addr in dataset:
            torch.save({"data": data, "label": label, "address": addr},
                       cache_dir / f"{addr[:10]}.pt")

    train_set, val_set, test_set = split_dataset(dataset)
    log.info("Split: train=%d  val=%d  test=%d", len(train_set), len(val_set), len(test_set))

    # ── model ─────────────────────────────────────────────────────────────────
    model = HTGNN(
        metadata=TRAIN_METADATA,
        hidden_dim=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_auc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_set, optimizer, device)
        val_auc, val_f1, _, _ = evaluate(model, val_set, device)
        scheduler.step()

        elapsed = time.time() - t0
        log.info("Epoch %3d/%d  loss=%.4f  val_auc=%.4f  val_f1=%.4f  (%.1fs)",
                 epoch, args.epochs, train_loss, val_auc, val_f1, elapsed)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), output_path)
            log.info("  ✓ Saved new best model (AUC=%.4f) → %s", best_auc, output_path)

    # ── final test evaluation ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(output_path, weights_only=True))
    log.info("\n=== Test Results ===")
    if test_set:
        test_auc, test_f1, probs, preds = evaluate(model, test_set, device)
        true_labels = [d[1] for d in test_set]
        test_acc = accuracy_score(true_labels, preds)
        test_cm = confusion_matrix(true_labels, preds, labels=[0, 1])
        log.info("AUC: %.4f   F1: %.4f", test_auc, test_f1)
        log.info("Accuracy: %.4f", test_acc)
        log.info("Confusion matrix [[TN, FP], [FN, TP]]:\n%s", test_cm)
        log.info("\n%s", classification_report(
            true_labels,
            preds,
            labels=[0, 1],
            target_names=["Benign", "Phishing"],
            zero_division=0,
        ))
    else:
        log.info("No test samples available after splitting; skipping final classification report.")
    log.info("Saved best model → %s", output_path)


if __name__ == "__main__":
    main()
