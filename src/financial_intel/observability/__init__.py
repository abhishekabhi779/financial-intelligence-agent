"""
Channel Intelligence Agent — Observability

LangSmith tracing, Prometheus metrics, cost tracking, structured logging.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from datetime import datetime
from functools import wraps

from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from financial_intel.config import get_settings


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)

# Agent metrics
agent_invocations = Counter(
    "agent_invocations_total",
    "Total agent invocations",
    ["agent", "status"],
)

agent_duration = Histogram(
    "agent_duration_seconds",
    "Agent execution duration",
    ["agent"],
)

# Token/Cost metrics
token_usage = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["model", "type"],  # prompt, completion
)

estimated_cost = Counter(
    "llm_cost_usd_total",
    "Estimated LLM cost in USD",
    ["model"],
)

# RAG metrics
rag_retrievals = Counter(
    "rag_retrievals_total",
    "Total RAG retrievals",
    ["search_type", "status"],
)

rag_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval duration",
    ["search_type"],
)

# Active requests gauge
active_requests = Gauge(
    "http_active_requests",
    "Currently active HTTP requests",
)


# ============================================================================
# LangSmith Tracing
# ============================================================================

def init_langsmith():
    """Initialize LangSmith tracing."""
    settings = get_settings()
    ls_config = settings.observability.langsmith

    if not ls_config.get("enabled", True):
        logger.info("LangSmith tracing disabled")
        return

    api_key = os.getenv("LANGSMITH_API_KEY") or ls_config.get("api_key")
    if not api_key:
        logger.warning("LangSmith API key not configured")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = ls_config.get("project_name", "channel-intel")

    # Sampling rate
    sample_rate = ls_config.get("trace_sample_rate", 1.0)
    if sample_rate < 1.0:
        os.environ["LANGCHAIN_SAMPLING_RATE"] = str(sample_rate)

    logger.info(f"LangSmith tracing initialized: project={ls_config.get('project_name')}")


def trace_agent(agent_name: str):
    """Decorator to trace agent execution with LangSmith."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            settings = get_settings()
            if not settings.observability.langsmith.get("enabled", True):
                return await func(*args, **kwargs)

            try:
                from langsmith import traceable
                traced_func = traceable(name=f"agent_{agent_name}")(func)
                return await traced_func(*args, **kwargs)
            except ImportError:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Cost Tracking
# ============================================================================

class CostTracker:
    """Track LLM token usage and estimated costs."""

    # Approximate pricing per 1K tokens (update as needed)
    PRICING = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015},
        "gemini-1.5-pro": {"prompt": 0.00125, "completion": 0.005},
        "text-embedding-3-small": {"prompt": 0.00002, "completion": 0},
        "text-embedding-3-large": {"prompt": 0.00013, "completion": 0},
    }

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.observability.cost_tracking.get("enabled", True)
        self.daily_budget = self.settings.observability.cost_tracking.get("budget_usd_per_day", 50.0)
        self.alert_threshold = self.settings.observability.cost_tracking.get("alert_threshold", 0.8)
        self._daily_cost = 0.0
        self._session_costs: Dict[str, float] = {}

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        pricing = self.PRICING.get(model, {"prompt": 0.001, "completion": 0.002})
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        return prompt_cost + completion_cost

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        session_id: Optional[str] = None,
    ) -> float:
        """Record token usage and return cost."""
        if not self.enabled:
            return 0.0

        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        # Update Prometheus metrics
        token_usage.labels(model=model, type="prompt").inc(prompt_tokens)
        token_usage.labels(model=model, type="completion").inc(completion_tokens)
        estimated_cost.labels(model=model).inc(cost)

        # Track daily budget
        self._daily_cost += cost
        if session_id:
            self._session_costs[session_id] = self._session_costs.get(session_id, 0) + cost

        # Check budget alert
        if self._daily_cost >= self.daily_budget * self.alert_threshold:
            logger.warning(f"Daily cost alert: ${self._daily_cost:.2f} / ${self.daily_budget:.2f}")

        return cost

    def get_daily_cost(self) -> float:
        return self._daily_cost

    def get_session_cost(self, session_id: str) -> float:
        return self._session_costs.get(session_id, 0.0)

    def check_budget(self) -> Dict[str, Any]:
        return {
            "daily_cost": self._daily_cost,
            "daily_budget": self.daily_budget,
            "utilization": self._daily_cost / self.daily_budget if self.daily_budget > 0 else 0,
            "alert_triggered": self._daily_cost >= self.daily_budget * self.alert_threshold,
        }

    def reset_daily(self):
        self._daily_cost = 0.0
        self._session_costs.clear()


cost_tracker = CostTracker()


# ============================================================================
# Structured Logging
# ============================================================================

def init_logging():
    """Initialize structured logging with loguru."""
    settings = get_settings()
    log_config = settings.observability.logging

    # Remove default handler
    logger.remove()

    # Console handler
    log_format = log_config.get("format", "json")
    if log_format == "json":
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            level=log_config.get("level", "INFO"),
            serialize=True,
        )
    else:
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_config.get("level", "INFO"),
        )

    # File handler
    log_file = log_config.get("file", "./logs/financial_intel.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        sink=log_file,
        rotation=log_config.get("rotation", "10 MB"),
        retention=log_config.get("retention", "7 days"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        level=log_config.get("level", "INFO"),
        serialize=log_format == "json",
    )

    logger.info("Logging initialized")


# ============================================================================
# FastAPI Middleware
# ============================================================================

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for observability."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        active_requests.inc()

        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

        # Log request
        logger.info(
            "HTTP request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)

            # Record metrics
            duration = time.perf_counter() - start_time
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()
            http_request_duration.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)

            logger.info(
                "HTTP request completed",
                request_id=request_id,
                status=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.4f}"

            return response

        except Exception as e:
            duration = time.perf_counter() - start_time
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500,
            ).inc()

            logger.exception(
                "HTTP request failed",
                request_id=request_id,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
            )
            raise

        finally:
            active_requests.dec()


# ============================================================================
# Health Check
# ============================================================================

async def health_check() -> Dict[str, Any]:
    """Comprehensive health check."""
    from financial_intel.rag.pipeline import get_rag_pipeline
    from financial_intel.tools.registry import get_registry

    checks = {}

    # RAG pipeline
    try:
        rag = get_rag_pipeline()
        stats = rag.get_collection_stats()
        checks["rag"] = {"status": "healthy", "details": stats}
    except Exception as e:
        checks["rag"] = {"status": "unhealthy", "error": str(e)}

    # Tool registry
    try:
        registry = get_registry()
        tools = registry.list_tools()
        checks["tools"] = {"status": "healthy", "count": len(tools)}
    except Exception as e:
        checks["tools"] = {"status": "unhealthy", "error": str(e)}

    # Cost tracker
    budget = cost_tracker.check_budget()
    checks["cost"] = {"status": "healthy" if not budget["alert_triggered"] else "warning", "details": budget}

    # Overall status
    overall = "healthy"
    if any(c["status"] == "unhealthy" for c in checks.values()):
        overall = "unhealthy"
    elif any(c["status"] == "warning" for c in checks.values()):
        overall = "degraded"

    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
    }