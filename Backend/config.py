"""
chainguard/backend/config.py
Centralised settings loaded from environment / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Alchemy
    alchemy_api_key: str = ""
    alchemy_mainnet_url: str = ""
    alchemy_ws_url: str = ""

    # Etherscan
    etherscan_api_key: str = ""
    etherscan_base_url: str = "https://api.etherscan.io/v2/api"

    # Model
    model_path: str = "./model/htgnn_phishing.pt"
    model_device: str = "cpu"
    subgraph_hop_depth: int = 2
    max_neighbors: int = 30

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # App
    cors_origins: str = (
        "http://localhost:3000,"
        "http://127.0.0.1:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:5173,"
        "http://localhost:8501,"
        "http://127.0.0.1:8501"
    )
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
