"""""
Endpoints
---------
  GET  /health                    – liveness probe
  POST /api/v1/analyze            – full phishing analysis for an address
  GET  /api/v1/analyze/{address}  – same via GET (for browser / curl)
  GET  /api/v1/subgraph/{address} – raw subgraph data (debug)
  GET  /api/v1/marketplace        – community validation listings
  POST /api/v1/marketplace        – submit a new validation report
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import get_settings
from pipeline.fetcher import EthereumFetcher
from pipeline.graph_builder import HeteroGraphBuilder
from model.htgnn import get_model
from model.inference import run_inference, PhishingResult
from api.cache import get_cache, set_cache
from api.marketplace import router as marketplace_router
from api.validators import validate_eth_address

log = logging.getLogger(__name__)
cfg = get_settings()

# ── startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load model on startup (warm cache)
    log.info("Loading HT-GNN model …")
    get_model()
    log.info("Model ready. ChainGuard API started.")
    yield
    log.info("ChainGuard API shutting down.")


# ── app factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ChainGuard – Ethereum Phishing Detector",
    description="HT-GNN powered phishing wallet detection via transaction graph analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(marketplace_router, prefix="/api/v1/marketplace")


# ── request / response schemas ────────────────────────────────────────────────

class FetchAndSaveRequest(BaseModel):
    addresses: list[str] = Field(..., description="List of Ethereum addresses to fetch")
    output_file: str = Field(default="./data/fetched_wallet_data.csv", 
                             description="Output CSV file path")
    save_raw_json: bool = Field(False, description="Also save raw JSON data per address")


class FetchAndSaveResponse(BaseModel):
    success: bool
    output_file: str
    addresses_processed: int
    addresses_failed: int
    rows_saved: int
    timestamp: str
    errors: list[str] = []


class FetchRawRequest(BaseModel):
    address: str = Field(..., description="Ethereum address to fetch")


class FetchRawResponse(BaseModel):
    address: str
    timestamp: str
    target: dict[str, Any]
    hop1: list[dict[str, Any]]
    hop2: list[dict[str, Any]]


class AnalyzeRequest(BaseModel):
    address: str = Field(..., description="Ethereum address (0x…) or ENS name")
    force_refresh: bool = Field(False, description="Bypass cache and re-fetch")


class EvidenceItemOut(BaseModel):
    kind: str
    label: str
    text: str
    attention_weight: float


class FeatureImportanceOut(BaseModel):
    name: str
    value: float
    raw: float


class SuspiciousNeighborOut(BaseModel):
    address: str
    tx_count: int
    balance_eth: float
    reason: str


class SubgraphNodeOut(BaseModel):
    id: str
    address: str
    node_type: str      # "eoa" | "contract"
    hop: int
    is_target: bool


class SubgraphEdgeOut(BaseModel):
    source: str
    target: str
    edge_type: str
    value: float


class AnalyzeResponse(BaseModel):
    address: str
    phishing_score: int
    confidence: float
    verdict: str
    risk_tier: str
    num_hop1_neighbors: int
    num_contracts: int
    num_inflow_txs: int
    num_approvals: int
    raw_phishing_prob: float
    model_used: str
    explanation: str
    evidence: list[EvidenceItemOut]
    feature_importance: list[FeatureImportanceOut]
    suspicious_neighbors: list[SuspiciousNeighborOut] = []
    inference_time_ms: float
    fetch_time_s: float
    subgraph: dict[str, Any]
    cached: bool


# ── core analysis logic ───────────────────────────────────────────────────────

async def analyse_address(address: str, force_refresh: bool = False) -> AnalyzeResponse:
   
    address = validate_eth_address(address)

    # Cache check
    cache_key = f"analysis:{address}"
    if not force_refresh:
        cached = await get_cache(cache_key)
        if cached:
            cached["cached"] = True
            return AnalyzeResponse(**cached)

    # Fetch on-chain data
    t0 = time.time()
    async with EthereumFetcher() as fetcher:
        raw = await fetcher.collect_full_subgraph_data(address)

    # Build heterogeneous graph
    builder = HeteroGraphBuilder()
    data, meta = builder.build(raw)

    # Run inference
    model = get_model()
    result: PhishingResult = run_inference(data, meta, raw, model)
    suspicious_neighbors = _extract_suspicious_neighbors(raw)

    # Build subgraph for visualisation
    subgraph = _build_subgraph_payload(raw, meta)

    # Compose response
    resp_dict = {
        "address":            result.address,
        "phishing_score":     result.phishing_score,
        "confidence":         result.confidence,
        "verdict":            result.verdict,
        "risk_tier":          result.risk_tier,
        "num_hop1_neighbors": result.num_hop1_neighbors,
        "num_contracts":      result.num_contracts,
        "num_inflow_txs":     result.num_inflow_txs,
        "num_approvals":      result.num_approvals,
        "raw_phishing_prob":  result.raw_phishing_prob,
        "model_used":         result.model_used,
        "explanation":        result.explanation,
        "evidence": [
            {"kind": e.kind, "label": e.label, "text": e.text,
             "attention_weight": e.attention_weight}
            for e in result.evidence
        ],
        "feature_importance": [
            {"name": f.name, "value": f.value, "raw": f.raw}
            for f in result.feature_importance
        ],
        "suspicious_neighbors": suspicious_neighbors,
        "inference_time_ms":  result.inference_time_ms,
        "fetch_time_s":       result.fetch_time_s,
        "subgraph":           subgraph,
        "cached":             False,
    }

    # Store in cache
    await set_cache(cache_key, resp_dict, ttl=cfg.cache_ttl_seconds)
    return AnalyzeResponse(**resp_dict)


def _extract_suspicious_neighbors(raw: dict, limit: int = 10) -> list[dict[str, Any]]:
    """Return suspicious hop-1 neighbors for dashboard rendering."""
    out: list[dict[str, Any]] = []
    for node in raw.get("hop1", []):
        balance = float(node.get("balance_eth", 0.0))
        tx_count = int(node.get("tx_count", 0))
        sampled_tx_count = int(node.get("sampled_tx_count", 0))
        is_suspicious = balance < 0.01 and (tx_count >= 50 or sampled_tx_count >= 20)
        if is_suspicious:
            activity_count = max(tx_count, sampled_tx_count)
            out.append({
                "address": node.get("address", ""),
                "tx_count": activity_count,
                "balance_eth": round(balance, 6),
                "reason": "Very low balance with unusually high observed activity",
            })
    out.sort(key=lambda n: n["tx_count"], reverse=True)
    return out[:limit]


def _build_subgraph_payload(raw: dict, meta: dict) -> dict:
    """Builds a D3-friendly {nodes, edges} payload for the frontend graph."""
    nodes: list[dict] = []
    edges: list[dict] = []

    target_addr = raw["target"]["address"]

    nodes.append({
        "id": target_addr, "address": target_addr,
        "node_type": "contract" if raw["target"]["is_contract"] else "eoa",
        "hop": 0, "is_target": True,
        "balance_eth": round(raw["target"]["balance_eth"], 4),
    })
    for h1 in raw.get("hop1", []):
        addr = h1["address"]
        nodes.append({
            "id": addr, "address": addr,
            "node_type": "contract" if h1.get("is_contract") else "eoa",
            "hop": 1, "is_target": False,
            "balance_eth": round(h1.get("balance_eth", 0), 4),
        })
        # Edge from target to hop-1
        for tx in (h1.get("txs") or [])[:3]:
            val = int(tx.get("value", 0)) / 1e18
            frm = tx.get("from", "").lower()
            to_ = tx.get("to", "").lower()
            if frm in (target_addr, addr) and to_ in (target_addr, addr):
                edges.append({
                    "source": frm, "target": to_,
                    "edge_type": "sends", "value": round(val, 6)
                })
    for h2 in raw.get("hop2", [])[:20]:
        addr = h2["address"]
        parent = h2.get("parent", target_addr)
        nodes.append({
            "id": addr, "address": addr,
            "node_type": "eoa", "hop": 2, "is_target": False,
            "balance_eth": 0,
        })
        edges.append({
            "source": parent, "target": addr,
            "edge_type": "sends", "value": 0
        })

    # Deduplicate nodes
    seen = set()
    unique_nodes = []
    for n in nodes:
        if n["id"] not in seen:
            seen.add(n["id"])
            unique_nodes.append(n)

    return {"nodes": unique_nodes, "edges": edges}


# ── bulk fetch and save logic ────────────────────────────────────────────────

async def fetch_and_save_worker(
    addresses: list[str],
    output_file: str,
    save_raw_json: bool = False
) -> FetchAndSaveResponse:
    """
    Fetch transaction data from Etherscan/Alchemy and save to CSV.
    Returns summary statistics.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    processed = 0

    async with EthereumFetcher() as fetcher:
        for addr in addresses:
            try:
                addr = validate_eth_address(addr)
                raw = await fetcher.collect_full_subgraph_data(addr)

                # Extract key stats for CSV
                target = raw.get("target", {})
                hop1_count = len(raw.get("hop1", []))
                hop2_count = len(raw.get("hop2", []))

                row = {
                    "address": addr,
                    "timestamp": datetime.utcnow().isoformat(),
                    "is_contract": target.get("is_contract", False),
                    "balance_eth": round(float(target.get("balance_eth", 0)), 6),
                    "first_block": target.get("first_block", 0),
                    "tx_count": int(target.get("tx_count", 0)),
                    "hop1_neighbors": hop1_count,
                    "hop2_neighbors": hop2_count,
                    "normal_txs": len(target.get("normal_txs", [])),
                    "internal_txs": len(target.get("internal_txs", [])),
                    "token_txs": len(target.get("token_txs", [])),
                    "approvals": len(target.get("approvals", [])),
                }
                rows.append(row)
                processed += 1

                # Optionally save raw JSON per address
                if save_raw_json:
                    json_path = output_path.parent / f"{addr}_raw.json"
                    with open(json_path, "w") as f:
                        json.dump(raw, f, indent=2, default=str)

                log.info(f"Fetched {addr}: {hop1_count} hop1, {hop2_count} hop2")

            except Exception as e:
                err_msg = f"{addr}: {str(e)}"
                errors.append(err_msg)
                log.warning(f"Fetch failed for {addr}: {e}")

    # Write CSV
    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return FetchAndSaveResponse(
        success=len(errors) == 0 or processed > 0,
        output_file=str(output_path),
        addresses_processed=processed,
        addresses_failed=len(addresses) - processed,
        rows_saved=len(rows),
        timestamp=datetime.utcnow().isoformat(),
        errors=errors[:10],  # Cap at 10 errors in response
    )


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.model_path}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_post(req: AnalyzeRequest):
    try:
        return await analyse_address(req.address, req.force_refresh)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Analysis failed for %s", req.address)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.get("/api/v1/analyze/{address}", response_model=AnalyzeResponse)
async def analyze_get(address: str, force_refresh: bool = False):
    try:
        return await analyse_address(address, force_refresh)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Analysis failed for %s", address)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.get("/api/v1/subgraph/{address}")
async def subgraph_raw(address: str):
    """Debug endpoint returning raw subgraph structure."""
    address = validate_eth_address(address)
    async with EthereumFetcher() as fetcher:
        raw = await fetcher.collect_full_subgraph_data(address)
    builder = HeteroGraphBuilder()
    _, meta = builder.build(raw)
    return {"meta": meta, "hop1_count": len(raw["hop1"]), "hop2_count": len(raw["hop2"])}


@app.post("/api/v1/fetch-and-save", response_model=FetchAndSaveResponse)
async def fetch_and_save_data(req: FetchAndSaveRequest):
    """
    Fetch transaction data from Etherscan + Alchemy for a list of addresses
    and save to a CSV file. Returns statistics.

    Example request:
    {
      "addresses": ["0x...", "0x..."],
      "output_file": "./data/wallet_data.csv",
      "save_raw_json": false
    }
    """
    if not req.addresses:
        raise HTTPException(status_code=400, detail="addresses list cannot be empty")
    if len(req.addresses) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 addresses per request"
        )

    try:
        result = await fetch_and_save_worker(
            req.addresses,
            req.output_file,
            req.save_raw_json
        )
        return result
    except Exception as e:
        log.exception("Fetch and save failed")
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


@app.post("/api/v1/fetch-raw", response_model=FetchRawResponse)
async def fetch_raw_data(req: FetchRawRequest):
    """
    Fetch raw transaction data from Etherscan + Alchemy for a single address.
    Returns target, hop1, and hop2 transaction data.

    Example request:
    {
      "address": "0x..."
    }
    """
    try:
        address = validate_eth_address(req.address)
        async with EthereumFetcher() as fetcher:
            raw = await fetcher.collect_full_subgraph_data(address)

        return FetchRawResponse(
            address=address,
            timestamp=datetime.utcnow().isoformat(),
            target=raw.get("target", {}),
            hop1=raw.get("hop1", [])[:100],  # Limit to 100 for response size
            hop2=raw.get("hop2", [])[:100],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Raw fetch failed for %s", req.address)
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=cfg.log_level)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
