"""
chainguard/backend/api/marketplace.py

Community validation marketplace endpoints.
In production this would be backed by PostgreSQL / MongoDB.
For now, uses an in-memory store that resets on restart.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["Marketplace"])

# ── in-memory store ───────────────────────────────────────────────────────────
_STORE: list[dict] = [
    {
        "id": "m1", "address": "0xa090e606e30bd747d4e6245a1517ebe430f0057e",
        "title": "Inferno Drainer Campaign", "author": "0x...a1f4",
        "tags": ["drainer", "nft-sweep", "high-risk"], "phishing_score": 97,
        "votes": 142, "verified": True, "created_at": int(time.time()) - 7200,
        "description": "Classic drainer pattern: 47 micro-inflows then bulk drain via approved contract.",
    },
    {
        "id": "m2", "address": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
        "title": "Verified: Uniswap V2 Router", "author": "0x...b3e2",
        "tags": ["verified-dex", "safe", "uniswap"], "phishing_score": 3,
        "votes": 890, "verified": True, "created_at": int(time.time()) - 86400,
        "description": "Widely-used DEX router. High volume is expected behaviour.",
    },
    {
        "id": "m3", "address": "0xfd6cc4f251d4a8cf004649db7a3e8db8e94c3e7a",
        "title": "Suspicious Airdrop Collector", "author": "0x...c9d1",
        "tags": ["suspicious", "airdrop", "monitor"], "phishing_score": 74,
        "votes": 67, "verified": False, "created_at": int(time.time()) - 14400,
        "description": "Elevated inflow variance and unverified contract approval present.",
    },
    {
        "id": "m4", "address": "0xb3f539dc",
        "title": "Wallet Drainer via Permit2", "author": "0x...f2a9",
        "tags": ["permit2", "drainer", "signature-phish"], "phishing_score": 99,
        "votes": 218, "verified": True, "created_at": int(time.time()) - 1800,
        "description": "Exploits Permit2 signature phishing. Immediate risk.",
    },
    {
        "id": "m5", "address": "0x87870bca3f3fd91a4fcef8fc4a9df5efb3b27b34",
        "title": "Aave V3 Pool (Verified)", "author": "0x...11bb",
        "tags": ["verified", "lending", "aave"], "phishing_score": 1,
        "votes": 1203, "verified": True, "created_at": int(time.time()) - 432000,
        "description": "Aave V3 lending pool. Safe for interaction.",
    },
    {
        "id": "m6", "address": "0x442198fa",
        "title": "New Honeypot Token Contract", "author": "0x...d5c3",
        "tags": ["honeypot", "token", "erc20"], "phishing_score": 81,
        "votes": 41, "verified": False, "created_at": int(time.time()) - 3600,
        "description": "Honeypot ERC-20 token – cannot sell after purchase.",
    },
]


# ── schemas ───────────────────────────────────────────────────────────────────

class MarketplaceItem(BaseModel):
    id: str
    address: str
    title: str
    author: str
    tags: list[str]
    phishing_score: int
    votes: int
    verified: bool
    created_at: int
    description: str


class SubmitReportRequest(BaseModel):
    address: str
    title: str = Field(..., max_length=120)
    author: str = Field(default="anonymous")
    tags: list[str] = Field(default_factory=list)
    phishing_score: int = Field(..., ge=0, le=100)
    description: str = Field(..., max_length=500)


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[MarketplaceItem])
async def list_reports(sort: str = "votes", limit: int = 20):
    """List community validation reports."""
    items = list(_STORE)
    if sort == "votes":
        items.sort(key=lambda x: x["votes"], reverse=True)
    elif sort == "recent":
        items.sort(key=lambda x: x["created_at"], reverse=True)
    elif sort == "score":
        items.sort(key=lambda x: x["phishing_score"], reverse=True)
    return items[:limit]


@router.post("", response_model=MarketplaceItem, status_code=201)
async def submit_report(req: SubmitReportRequest):
    """Submit a new validation report."""
    item = {
        "id": str(uuid.uuid4())[:8],
        "address": req.address.lower(),
        "title": req.title,
        "author": req.author,
        "tags": req.tags,
        "phishing_score": req.phishing_score,
        "votes": 1,
        "verified": False,
        "created_at": int(time.time()),
        "description": req.description,
    }
    _STORE.insert(0, item)
    return item


@router.post("/{report_id}/vote")
async def vote(report_id: str):
    """Upvote a report."""
    for item in _STORE:
        if item["id"] == report_id:
            item["votes"] += 1
            return {"votes": item["votes"]}
    raise HTTPException(status_code=404, detail="Report not found")
