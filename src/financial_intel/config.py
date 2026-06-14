"""
Channel Intelligence Agent — Configuration Management

Pydantic Settings with YAML + environment variable support.
All settings validated at startup.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMModelConfig(BaseSettings):
    """Individual LLM model configuration."""

    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3


class LLMConfig(BaseSettings):
    """LLM configuration."""

    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3
    models: Dict[str, LLMModelConfig] = {}
    agent_models: Dict[str, str] = {}


class EmbeddingsConfig(BaseSettings):
    """Embedding model configuration."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""

    provider: Literal["chromadb", "pinecone", "weaviate"] = "chromadb"
    chromadb: Dict[str, Any] = Field(default_factory=dict)
    pinecone: Dict[str, Any] = Field(default_factory=dict)
    weaviate: Dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseSettings):
    """RAG pipeline configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: Literal["fixed", "semantic", "recursive"] = "semantic"
    semantic_chunker_threshold: float = 0.7
    top_k: int = 10
    rerank_top_k: int = 5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_search_alpha: float = 0.5
    similarity_threshold: float = 0.6


class ToolConfig(BaseSettings):
    """Tool configuration."""

    web_search: Dict[str, Any] = Field(default_factory=dict)
    news: Dict[str, Any] = Field(default_factory=dict)
    vendor_api: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseSettings):
    """Agent-specific configuration."""

    supervisor: Dict[str, Any] = Field(default_factory=dict)
    filings_agent: Dict[str, Any] = Field(default_factory=dict)
    market_agent: Dict[str, Any] = Field(default_factory=dict)
    stakeholder_agent: Dict[str, Any] = Field(default_factory=dict)
    synthesis: Dict[str, Any] = Field(default_factory=dict)


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""

    langsmith: Dict[str, Any] = Field(default_factory=dict)
    prometheus: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    cost_tracking: Dict[str, Any] = Field(default_factory=dict)


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = Field(default_factory=list)
    rate_limit: Dict[str, int] = Field(default_factory=dict)
    auth: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""

    golden_set_path: str = "./configs/eval/golden_sets.jsonl"
    judges: Dict[str, Any] = Field(default_factory=dict)
    benchmarks: List[Dict[str, Any]] = Field(default_factory=list)


class PathsConfig(BaseSettings):
    """File system paths."""

    vendor_docs: str = "./data/vendor_docs"
    partner_data: str = "./data/partner_data"
    market_reports: str = "./data/market_reports"
    chromadb: str = "./data/chromadb"
    logs: str = "./logs"
    cache: str = "./cache"


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # App metadata
    name: str = "channel-intel"
    version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Sub-configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        if v not in ("development", "staging", "production"):
            raise ValueError("environment must be development, staging, or production")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        if v.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("log_level must be a valid logging level")
        return v.upper()


# Global settings instance
_settings: Optional[AppConfig] = None


def get_settings() -> AppConfig:
    """Get global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = AppConfig()
    return _settings


def load_settings_from_yaml(config_path: str = "configs/settings.yaml") -> AppConfig:
    """Load settings from YAML file with env var overrides."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        yaml_data = yaml.safe_load(f)

    # Flatten nested dict for Pydantic Settings
    def flatten_dict(d: dict, parent_key: str = "", sep: str = "__") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(yaml_data)
    return AppConfig(**flat_config)


# Convenience function for getting model config
def get_llm_config(agent_name: str = "default") -> LLMModelConfig:
    """Get LLM config for a specific agent."""
    settings = get_settings()
    model_key = settings.llm.agent_models.get(agent_name, settings.llm.default_model)
    return settings.llm.models.get(model_key, LLMModelConfig(
        provider=settings.llm.default_provider,
        model=settings.llm.default_model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    ))