from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from pipeline.fetcher import EthereumFetcher
from pipeline.graph_builder import HeteroGraphBuilder
from model.htgnn import load_model
from model.inference import run_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HT-GNN on forta_merged_test.csv")
    p.add_argument("--dataset", required=True, help="Path to test CSV with address column")
    p.add_argument("--model", required=True, help="Path to model checkpoint")
    p.add_argument("--limit", type=int, default=20, help="Number of test addresses to evaluate")
    p.add_argument("--out", default="./data/forta_test_predictions.csv", help="Output predictions CSV")
    return p.parse_args()


def load_addresses(csv_path: str, limit: int) -> list[str]:
    out: list[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = (row.get("address") or row.get("Address") or "").strip().lower()
            if not addr:
                continue
            out.append(addr)
            if limit > 0 and len(out) >= limit:
                break
    return out


async def main() -> None:
    args = parse_args()
    addresses = load_addresses(args.dataset, args.limit)
    if not addresses:
        print("No valid addresses found in test CSV")
        return

    model = load_model(args.model)
    builder = HeteroGraphBuilder()

    rows: list[dict] = []

    async with EthereumFetcher() as fetcher:
        for addr in addresses:
            try:
                raw = await fetcher.collect_full_subgraph_data(addr)
                data, meta = builder.build(raw)
                result = run_inference(data, meta, raw, model=model)
                rows.append({
                    "address": addr,
                    "phishing_score": result.phishing_score,
                    "raw_phishing_prob": result.raw_phishing_prob,
                    "verdict": result.verdict,
                    "risk_tier": result.risk_tier,
                    "confidence": result.confidence,
                })
                print(f"OK {addr} score={result.phishing_score} tier={result.risk_tier}")
            except Exception as e:
                rows.append({
                    "address": addr,
                    "phishing_score": "",
                    "raw_phishing_prob": "",
                    "verdict": "error",
                    "risk_tier": "error",
                    "confidence": "",
                    "error": str(e),
                })
                print(f"SKIP {addr}: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["address", "phishing_score", "raw_phishing_prob", "verdict", "risk_tier", "confidence", "error"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if "error" not in r:
                r["error"] = ""
            writer.writerow(r)

    print(f"Saved predictions: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
