"""
Channel Intelligence Agent — FastAPI Application

REST API with streaming responses, authentication, rate limiting.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from financial_intel.config import get_settings
from financial_intel.graph.supervisor import run_research
from financial_intel.observability import (
    init_langsmith,
    init_logging,
    ObservabilityMiddleware,
    health_check,
    cost_tracker,
)
from financial_intel.state import ChannelIntelState


# ============================================================================
# Request/Response Models
# ============================================================================

class ResearchRequest(BaseModel):
    """Research request payload."""

    query: str = Field(..., min_length=1, max_length=2000, description="Research query")
    session_id: Optional[str] = Field(None, description="Optional session ID")
    user_id: Optional[str] = Field(None, description="Optional user ID")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Config overrides")


class ResearchResponse(BaseModel):
    """Research response payload."""

    session_id: str
    query: str
    briefing: str
    opportunities: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    token_usage: Dict[str, Any]
    duration_seconds: float
    status: str


class StreamingEvent(BaseModel):
    """Server-sent event for streaming."""

    event: str
    data: Dict[str, Any]
    session_id: str


# ============================================================================
# Authentication
# ============================================================================

security = HTTPBearer(auto_error=False)


async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Verify API token if auth enabled."""
    settings = get_settings()
    if not settings.api.auth.get("enabled", False):
        return None

    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    expected_token = settings.api.auth.get("secret_key")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")

    return credentials.credentials


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    init_logging()
    init_langsmith()

    # Create required directories
    settings = get_settings()
    for path in [settings.paths.logs, settings.paths.cache, settings.paths.chromadb]:
        import os
        os.makedirs(path, exist_ok=True)

    yield

    # Shutdown
    logger.info("Application shutting down")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Channel Intelligence Agent",
    description="Multi-agent research agent for channel sales intelligence (LangGraph + AutoGen + CrewAI)",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(ObservabilityMiddleware)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (simple in-memory)
_request_counts: Dict[str, List[float]] = {}


async def rate_limit(request: Request):
    """Simple rate limiting middleware."""
    if not settings.api.rate_limit:
        return

    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now().timestamp()
    minute_ago = now - 60

    if client_ip not in _request_counts:
        _request_counts[client_ip] = []

    _request_counts[client_ip] = [t for t in _request_counts[client_ip] if t > minute_ago]

    if len(_request_counts[client_ip]) >= settings.api.rate_limit.get("requests_per_minute", 60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    _request_counts[client_ip].append(now)


# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return await health_check()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/research", response_model=ResearchResponse, dependencies=[Depends(rate_limit)])
async def research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    token: Optional[str] = Depends(verify_token),
):
    """
    Execute full research pipeline.

    Returns complete briefing with opportunities and citations.
    """
    start_time = datetime.now()
    session_id = request.session_id or str(uuid.uuid4())[:8]

    try:
        # Run research
        final_state: ChannelIntelState = await run_research(
            user_query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            config_overrides=request.config_overrides,
        )

        duration = (datetime.now() - start_time).total_seconds()

        return ResearchResponse(
            session_id=session_id,
            query=request.query,
            briefing=final_state.get("briefing_final", ""),
            opportunities=[opp.model_dump() for opp in final_state.get("opportunities", [])],
            citations=final_state.get("citations", []),
            token_usage=final_state.get("token_usage", {}).model_dump() if final_state.get("token_usage") else {},
            duration_seconds=duration,
            status="completed" if final_state.get("is_complete") else "partial",
        )

    except Exception as e:
        logger.exception(f"Research failed for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/stream", dependencies=[Depends(rate_limit)])
async def research_stream(
    request: ResearchRequest,
    token: Optional[str] = Depends(verify_token),
):
    """
    Stream research progress via Server-Sent Events.

    Events:
    - planning: Research plan created
    - vendor_research: Vendor research step
    - market_research: Market research step
    - partner_research: Partner research step
    - synthesis: Synthesis step
    - complete: Final result
    - error: Error occurred
    """
    session_id = request.session_id or str(uuid.uuid4())[:8]

    async def event_generator():
        try:
            # Initial event
            yield f"event: planning\ndata: {{\"session_id\": \"{session_id}\", \"status\": \"started\"}}\n\n"

            # Run with streaming (would need graph streaming support)
            final_state = await run_research(
                user_query=request.query,
                session_id=session_id,
                user_id=request.user_id,
                config_overrides=request.config_overrides,
            )

            # Stream step completion events
            steps = ["vendor_research", "market_research", "partner_research", "synthesis"]
            for step in steps:
                yield f"event: {step}\ndata: {{\"session_id\": \"{session_id}\", \"status\": \"completed\"}}\n\n"

            # Final result
            result_data = {
                "session_id": session_id,
                "query": request.query,
                "briefing": final_state.get("briefing_final", ""),
                "opportunities": [opp.model_dump() for opp in final_state.get("opportunities", [])],
                "citations": final_state.get("citations", []),
                "token_usage": final_state.get("token_usage", {}).model_dump() if final_state.get("token_usage") else {},
                "status": "completed",
            }
            import json
            yield f"event: complete\ndata: {json.dumps(result_data)}\n\n"

        except Exception as e:
            error_data = {"session_id": session_id, "error": str(e), "status": "error"}
            import json
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/research/{session_id}")
async def get_research_status(session_id: str, token: Optional[str] = Depends(verify_token)):
    """Get research status (would need persistent storage)."""
    # This would require a persistence layer (Redis, database)
    return {"session_id": session_id, "status": "not_implemented", "message": "Requires persistence layer"}


@app.post("/cost/check")
async def check_cost(token: Optional[str] = Depends(verify_token)):
    """Check current cost utilization."""
    return cost_tracker.check_budget()


@app.post("/cost/reset")
async def reset_cost(token: Optional[str] = Depends(verify_token)):
    """Reset daily cost tracker (admin only)."""
    cost_tracker.reset_daily()
    return {"status": "reset", "daily_cost": 0.0}


# ============================================================================
# Ingestion Endpoints (for building knowledge base)
# ============================================================================

class IngestVendorRequest(BaseModel):
    vendor_name: str
    doc_paths: List[str]


class IngestPartnerRequest(BaseModel):
    partner_name: str
    data: Dict[str, Any]


class IngestMarketRequest(BaseModel):
    report_name: str
    content: str
    metadata: Dict[str, Any] = {}


@app.post("/ingest/vendor")
async def ingest_vendor(request: IngestVendorRequest, token: Optional[str] = Depends(verify_token)):
    """Ingest vendor documentation."""
    from financial_intel.rag.pipeline import get_rag_pipeline

    rag = get_rag_pipeline()
    result = await rag.ingest_vendor_docs(request.vendor_name, request.doc_paths)
    return result


@app.post("/ingest/partner")
async def ingest_partner(request: IngestPartnerRequest, token: Optional[str] = Depends(verify_token)):
    """Ingest partner profile."""
    from financial_intel.rag.pipeline import get_rag_pipeline

    rag = get_rag_pipeline()
    result = await rag.ingest_partner_data(request.partner_name, request.data)
    return result


@app.post("/ingest/market")
async def ingest_market(request: IngestMarketRequest, token: Optional[str] = Depends(verify_token)):
    """Ingest market report."""
    from financial_intel.rag.pipeline import get_rag_pipeline

    rag = get_rag_pipeline()
    result = await rag.ingest_market_report(request.report_name, request.content, request.metadata)
    return result


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "financial_intel.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.environment == "development",
    )