"""
chainguard/backend/api/cache.py

Redis-backed async cache with transparent in-memory fallback.
Used to avoid redundant Etherscan API calls for recently scanned addresses.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# In-memory fallback: {key: (value, expiry_timestamp)}
_mem_cache: dict[str, tuple[Any, float]] = {}

_redis_client = None


async def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as aioredis
        from config import get_settings
        cfg = get_settings()
        _redis_client = aioredis.from_url(cfg.redis_url, decode_responses=True)
        await _redis_client.ping()
        log.info("Redis cache connected.")
        return _redis_client
    except Exception as e:
        log.warning("Redis unavailable (%s) – using in-memory cache.", e)
        return None


async def get_cache(key: str) -> Any | None:
    # Try Redis
    try:
        r = await _get_redis()
        if r:
            val = await r.get(key)
            if val:
                return json.loads(val)
    except Exception:
        pass

    # Fallback: in-memory
    if key in _mem_cache:
        val, expiry = _mem_cache[key]
        if time.time() < expiry:
            return val
        del _mem_cache[key]
    return None


async def set_cache(key: str, value: Any, ttl: int = 3600) -> None:
    # Try Redis
    try:
        r = await _get_redis()
        if r:
            await r.setex(key, ttl, json.dumps(value))
            return
    except Exception:
        pass

    # Fallback: in-memory
    _mem_cache[key] = (value, time.time() + ttl)
    # Evict old entries if cache grows too large
    if len(_mem_cache) > 500:
        now = time.time()
        expired = [k for k, (_, exp) in _mem_cache.items() if now > exp]
        for k in expired:
            del _mem_cache[k]


async def delete_cache(key: str) -> None:
    try:
        r = await _get_redis()
        if r:
            await r.delete(key)
    except Exception:
        pass
    _mem_cache.pop(key, None)
