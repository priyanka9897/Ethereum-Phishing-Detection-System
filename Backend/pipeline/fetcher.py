"""
chainguard/backend/pipeline/fetcher.py

Fetches raw Ethereum data from Etherscan and Alchemy RPC.
All network calls are async; results are returned as plain dicts
so the graph builder can stay framework-agnostic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

from config import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()

# Etherscan free tier is limited (~3 requests/sec). Keep a global paced gate
# to avoid bursty parallel calls causing widespread NOTOK rate-limit responses.
_ETHERSCAN_GATE = asyncio.Lock()
_ETHERSCAN_NEXT_TS = 0.0
_ETHERSCAN_MIN_INTERVAL_S = 0.50

# ── helpers ──────────────────────────────────────────────────────────────────

def _etherscan_url(**params) -> str:
    base = cfg.etherscan_base_url
    params.setdefault("chainid", 1)
    params["apikey"] = cfg.etherscan_api_key
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{query}"


async def _get(session: aiohttp.ClientSession, url: str) -> dict:
    global _ETHERSCAN_NEXT_TS
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            if url.startswith(cfg.etherscan_base_url):
                async with _ETHERSCAN_GATE:
                    now = time.monotonic()
                    wait_s = max(0.0, _ETHERSCAN_NEXT_TS - now)
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    _ETHERSCAN_NEXT_TS = max(now, _ETHERSCAN_NEXT_TS) + _ETHERSCAN_MIN_INTERVAL_S

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                r.raise_for_status()
                data = await r.json()

            # Etherscan can return HTTP 200 with NOTOK payloads.
            if isinstance(data, dict):
                status = str(data.get("status", ""))
                message = str(data.get("message", ""))
                result = data.get("result")

                if status == "0" and message.upper() == "NOTOK":
                    reason = str(result)
                    if "rate limit" in reason.lower() and attempt < 5:
                        await asyncio.sleep(1.2 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Etherscan error: {reason}")

            return data
        except Exception as e:
            last_err = e
            if attempt < 5:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                break

    raise RuntimeError(f"HTTP fetch failed after retries: {last_err}")


# ── public API ───────────────────────────────────────────────────────────────

class EthereumFetcher:
   
    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()
    # ── balance & basic info ─────────────────────────────────────────────────
    async def get_balance_eth(self, address: str) -> float:
        url = _etherscan_url(module="account", action="balance",
                             address=address, tag="latest")
        data = await _get(self._session, url)
        try:
            wei = int(data.get("result", 0))
        except (TypeError, ValueError):
            return 0.0
        return wei / 1e18

    async def get_account_age(self, address: str) -> int:
        """Returns block number of first seen transaction (0 if unknown)."""
        url = _etherscan_url(module="account", action="txlist",
                             address=address, startblock=0, endblock=99999999,
                             page=1, offset=1, sort="asc")
        data = await _get(self._session, url)
        txs = data.get("result", [])
        if isinstance(txs, list) and txs:
            return int(txs[0].get("blockNumber", 0))
        return 0

    async def is_contract(self, address: str) -> bool:
        url = _etherscan_url(module="proxy", action="eth_getCode",
                             address=address, tag="latest")
        data = await _get(self._session, url)
        code = data.get("result", "0x")
        return isinstance(code, str) and code.startswith("0x") and code not in ("0x", "0x0")

    # ── transactions ─────────────────────────────────────────────────────────

    async def get_normal_txs(self, address: str, limit: int = 100) -> list[dict]:
        url = _etherscan_url(module="account", action="txlist",
                             address=address, startblock=0, endblock=99999999,
                             page=1, offset=limit, sort="desc")
        data = await _get(self._session, url)
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    async def get_internal_txs(self, address: str, limit: int = 50) -> list[dict]:
        url = _etherscan_url(module="account", action="txlistinternal",
                             address=address, startblock=0, endblock=99999999,
                             page=1, offset=limit, sort="desc")
        data = await _get(self._session, url)
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    async def get_token_transfers(self, address: str, limit: int = 100) -> list[dict]:
        url = _etherscan_url(module="account", action="tokentx",
                             address=address, startblock=0, endblock=99999999,
                             page=1, offset=limit, sort="desc")
        data = await _get(self._session, url)
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    async def get_erc20_approvals(self, address: str, limit: int = 50) -> list[dict]:
        """Fetch ERC-20 Approval events (potential drainer authorisations)."""
        # Use Etherscan logs API for Approval(address,address,uint256)
        approval_topic = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"
        try:
            url = _etherscan_url(
                module="logs",
                action="getLogs",
                topic0=approval_topic,
                topic1="0x000000000000000000000000" + address.lower()[2:],
                fromBlock=0,
                toBlock="latest",
                page=1,
                offset=limit,
            )
            data = await _get(self._session, url)
            result = data.get("result", [])
            return result if isinstance(result, list) else []
        except Exception as e:
            # Approval logs are an auxiliary signal; do not fail full analysis.
            log.warning("approval log fetch failed for %s: %s", address, e)
            return []

    # ── Alchemy RPC extras ───────────────────────────────────────────────────

    async def alchemy_rpc(self, method: str, params: list) -> Any:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                async with self._session.post(
                    cfg.alchemy_mainnet_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as r:
                    r.raise_for_status()
                    data = await r.json()

                if isinstance(data, dict) and data.get("error"):
                    err = data["error"]
                    raise RuntimeError(f"Alchemy RPC error: {err}")

                return data.get("result")
            except Exception as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(0.7 * (attempt + 1))
                else:
                    break

        raise RuntimeError(f"Alchemy RPC failed after retries: {last_err}")

    async def get_transaction_count(self, address: str) -> int:
        result = await self.alchemy_rpc("eth_getTransactionCount", [address, "latest"])
        try:
            return int(result, 16) if result else 0
        except (TypeError, ValueError):
            return 0

    async def get_latest_block(self) -> int:
        result = await self.alchemy_rpc("eth_blockNumber", [])
        try:
            return int(result, 16) if result else 0
        except (TypeError, ValueError):
            return 0

      # ── batch neighbour collection ────────────────────────────────────────────
        """
        Return all unique addresses that have interacted with `address`
        in normal + internal + token transfer transactions (up to max_neighbors).
        """
    async def collect_neighbors(self, address: str) -> list[str]:
      
        normal, internal, tokens = await asyncio.gather(
            self.get_normal_txs(address, limit=cfg.max_neighbors),
            self.get_internal_txs(address, limit=cfg.max_neighbors // 2),
            self.get_token_transfers(address, limit=cfg.max_neighbors),
        )

        neighbors: set[str] = set()
        for tx in normal + internal + tokens:
            frm = tx.get("from", "").lower()
            to = tx.get("to", "").lower()
            if frm and frm != address.lower():
                neighbors.add(frm)
            if to and to != address.lower():
                neighbors.add(to)

        # Limit to max_neighbors
        return list(neighbors)[: cfg.max_neighbors]
        """
        Entry point: collects all data needed by the graph builder.
        Returns a structured dict with target info + hop-1 + hop-2 neighbours.
        """
    async def collect_full_subgraph_data(self, address: str) -> dict:
        
        t0 = time.time()
        address = address.lower()
        # ── Target node ───────────────────────────────────────────────────────
        (balance, is_contract, nonce, normal_txs, internal_txs,
         token_txs, approvals, first_block) = await asyncio.gather(
            self.get_balance_eth(address),
            self.is_contract(address),
            self.get_transaction_count(address),
            self.get_normal_txs(address),
            self.get_internal_txs(address),
            self.get_token_transfers(address),
            self.get_erc20_approvals(address),
            self.get_account_age(address),
        )
        hop1_addrs = await self.collect_neighbors(address)
        hop1_data: list[dict] = []  # Fetch hop-1 node features in parallel (balance + contract check)
        hop2_data: list[dict] = []
        tasks = [
            asyncio.gather(
                self.get_balance_eth(n),
                self.is_contract(n),
                self.get_transaction_count(n),
                self.get_normal_txs(n, limit=30),
            )
            for n in hop1_addrs[:cfg.max_neighbors]
        ]
        hop1_results = await asyncio.gather(*tasks, return_exceptions=True)
        for addr, res in zip(hop1_addrs, hop1_results):
            if isinstance(res, Exception):
                log.warning("hop1 fetch failed for %s: %s", addr, res)
                continue
            bal, is_c, tx_count, txs = res
            hop1_data.append({
                "address": addr,
                "balance_eth": bal,
                "is_contract": is_c,
                "tx_count": tx_count,
                "sampled_tx_count": len(txs),
                "txs": txs,
            })
        hop2_tasks = []
        hop2_parents = []
        for h1 in hop1_data[:10]:           # limit to 10 hop-1 to stay fast
            hop2_tasks.append(self.collect_neighbors(h1["address"]))
            hop2_parents.append(h1["address"])

        hop2_neighbor_lists = await asyncio.gather(*hop2_tasks, return_exceptions=True)
        for parent, nb_list in zip(hop2_parents, hop2_neighbor_lists):
            if isinstance(nb_list, Exception):
                continue
            for nb in nb_list[:5]:          # max 5 hop-2 per hop-1
                hop2_data.append({"address": nb, "parent": parent})

        elapsed = round(time.time() - t0, 2)
        log.info("Subgraph collected for %s in %.2fs", address, elapsed)

        return {
            "target": {
                "address": address,
                "balance_eth": balance,
                "is_contract": is_contract,
                "nonce": nonce,
                "normal_txs": normal_txs,
                "internal_txs": internal_txs,
                "token_txs": token_txs,
                "approvals": approvals,
                "first_block": first_block,
            },
            "hop1": hop1_data,
            "hop2": hop2_data,
            "fetch_time_s": elapsed,
        }
