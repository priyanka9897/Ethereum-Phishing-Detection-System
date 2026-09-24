"""
chainguard/backend/api/validators.py
"""
import re


ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
ENS_RE         = re.compile(r"^[a-zA-Z0-9\-]+\.eth$")


def validate_eth_address(address: str) -> str:
    """
    Validates and normalises an Ethereum address or ENS name.
    Returns the checksummed / lower-cased address.
    Raises ValueError for invalid inputs.
    """
    address = address.strip()

    if ENS_RE.match(address):
        # ENS resolution would require a Web3 call; we return as-is
        # and let the fetcher resolve it. Flag it clearly.
        return address.lower()

    if not ETH_ADDRESS_RE.match(address):
        raise ValueError(
            f"Invalid Ethereum address: '{address}'. "
            "Expected 0x followed by 40 hex characters, or a valid ENS name."
        )

    return address.lower()
