"""
Channel Intelligence Agent — LangGraph State Schema

TypedDict-based state for type-safe graph execution.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from datetime import datetime
from pydantic import BaseModel, Field


class VendorInfo(BaseModel):
    """Structured vendor information."""

    name: str
    description: str = ""
    product_categories: List[str] = Field(default_factory=list)
    pricing_model: str = ""
    target_markets: List[str] = Field(default_factory=list)
    partner_program: str = ""
    roadmap_highlights: List[str] = Field(default_factory=list)
    competitive_positioning: str = ""
    source_urls: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)


class MarketSignal(BaseModel):
    """Market intelligence signal."""

    signal_type: Literal["trend", "competitor_move", "earnings", "product_launch", "funding", "regulation"]
    title: str
    summary: str
    source: str
    source_url: str
    relevance_score: float
    timestamp: datetime
    entities: List[str] = Field(default_factory=list)
    sentiment: Literal["positive", "negative", "neutral"] = "neutral"


class PartnerProfile(BaseModel):
    """Channel partner profile."""

    company_name: str
    website: str = ""
    headquarters: str = ""
    employee_count: int = 0
    revenue_range: str = ""
    specializations: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    vendor_partnerships: List[str] = Field(default_factory=list)
    tech_stack: List[str] = Field(default_factory=list)
    buying_signals: List[str] = Field(default_factory=list)
    engagement_score: float = 0.0
    opportunity_score: float = 0.0
    last_contacted: Optional[datetime] = None
    source_urls: List[str] = Field(default_factory=list)


class Opportunity(BaseModel):
    """Scored channel opportunity."""

    id: str
    vendor: str
    partner: str
    opportunity_type: Literal["new_logo", "expansion", "renewal", "cross_sell", "tech_refresh"]
    score: float
    rationale: str
    vendor_fit: float
    partner_readiness: float
    market_timing: float
    estimated_value: Optional[float] = None
    next_actions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Research execution plan."""

    query: str
    vendor_focus: List[str] = Field(default_factory=list)
    market_focus: List[str] = Field(default_factory=list)
    partner_focus: List[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    max_vendors: int = 5
    max_partners: int = 10
    time_budget_seconds: int = 120


class TokenUsage(BaseModel):
    """Token usage tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


# LangGraph State (TypedDict for graph execution)
class ChannelIntelState(TypedDict):
    """
    Main state for the Channel Intelligence Research Agent.

    This TypedDict is used by LangGraph for type-safe state management
    across the supervisor graph and all sub-agents.
    """

    # Input
    user_query: str
    research_plan: Optional[ResearchPlan]
    session_id: str
    user_id: Optional[str]

    # Planning & Control
    current_step: str
    step_history: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    is_complete: bool
    error: Optional[str]

    # Agent Outputs
    vendor_research: Dict[str, VendorInfo]
    market_signals: List[MarketSignal]
    partner_profiles: Dict[str, PartnerProfile]
    opportunities: List[Opportunity]

    # Synthesis
    briefing_draft: str
    briefing_final: str
    citations: List[Dict[str, Any]]

    # Observability
    token_usage: TokenUsage
    agent_traces: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime]

    # Configuration
    config_overrides: Dict[str, Any]


# Sub-agent State Schemas
class VendorAgentState(TypedDict):
    """State for AutoGen Vendor Agent."""

    vendor_names: List[str]
    research_results: Dict[str, VendorInfo]
    messages: List[Dict[str, Any]]
    round_count: int
    max_rounds: int
    is_complete: bool


class MarketAgentState(TypedDict):
    """State for LangGraph Market Agent."""

    research_topics: List[str]
    signals_collected: List[MarketSignal]
    current_topic_index: int
    search_queries: List[str]
    raw_search_results: List[Dict[str, Any]]
    is_complete: bool
    error: Optional[str]


class PartnerAgentState(TypedDict):
    """State for CrewAI Partner Agent."""

    partner_names: List[str]
    profiles: Dict[str, PartnerProfile]
    tasks_completed: List[str]
    current_task: Optional[str]
    crew_output: Optional[str]
    is_complete: bool


class SynthesisState(TypedDict):
    """State for Synthesis Agent."""

    vendor_research: Dict[str, VendorInfo]
    market_signals: List[MarketSignal]
    partner_profiles: Dict[str, PartnerProfile]
    draft_briefing: str
    final_briefing: str
    opportunities: List[Opportunity]
    citations: List[Dict[str, Any]]
    is_complete: bool


# Helper functions
def create_initial_state(
    user_query: str,
    session_id: str,
    user_id: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ChannelIntelState:
    """Create initial state for a new research session."""
    from datetime import datetime

    settings = get_settings()

    return ChannelIntelState(
        user_query=user_query,
        research_plan=None,
        session_id=session_id,
        user_id=user_id,
        current_step="planning",
        step_history=[],
        iteration_count=0,
        max_iterations=settings.agents.supervisor.get("max_iterations", 10),
        is_complete=False,
        error=None,
        vendor_research={},
        market_signals=[],
        partner_profiles={},
        opportunities=[],
        briefing_draft="",
        briefing_final="",
        citations=[],
        token_usage=TokenUsage(),
        agent_traces=[],
        start_time=datetime.now(),
        end_time=None,
        config_overrides=config_overrides or {},
    )


def update_step(state: ChannelIntelState, step: str) -> ChannelIntelState:
    """Update current step and history."""
    from datetime import datetime

    state["step_history"].append({
        "step": state["current_step"],
        "timestamp": datetime.now().isoformat(),
        "iteration": state["iteration_count"],
    })
    state["current_step"] = step
    state["iteration_count"] += 1
    return state


def add_token_usage(state: ChannelIntelState, usage: TokenUsage) -> ChannelIntelState:
    """Accumulate token usage."""
    current = state["token_usage"]
    state["token_usage"] = TokenUsage(
        prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
        completion_tokens=current.completion_tokens + usage.completion_tokens,
        total_tokens=current.total_tokens + usage.total_tokens,
        estimated_cost_usd=current.estimated_cost_usd + usage.estimated_cost_usd,
    )
    return state