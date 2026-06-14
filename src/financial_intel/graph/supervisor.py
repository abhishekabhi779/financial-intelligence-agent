"""
Channel Intelligence Agent — LangGraph Supervisor Graph

Main orchestration graph that coordinates all specialist agents.
"""

from typing import Any, Dict, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from financial_intel.config import get_settings
from financial_intel.state import (
    ChannelIntelState,
    create_initial_state,
    update_step,
    add_token_usage,
    ResearchPlan,
    VendorInfo,
    MarketSignal,
    PartnerProfile,
    Opportunity,
    TokenUsage,
)
from financial_intel.graph.market_agent import run_market_agent
from financial_intel.graph.synthesis import run_synthesis
from financial_intel.agents.filings_agent import run_filings_agent
from financial_intel.agents.stakeholder_agent import run_stakeholder_agent


async def planning_node(state: ChannelIntelState, config: RunnableConfig) -> ChannelIntelState:
    """
    Planning node: Analyze user query and create research plan.
    Uses LLM to decompose query into vendor/market/partner research tasks.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from financial_intel.config import get_llm_config

    settings = get_settings()
    llm_config = get_llm_config("supervisor")

    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    ).with_structured_output(ResearchPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research planner for a channel intelligence agent.
Analyze the user's query and create a structured research plan.

Identify:
1. Which vendors to research (if mentioned or implied)
2. Which market topics/trends to investigate
3. Which partner segments to profile
4. Priority level and scope limits

Output a ResearchPlan with specific, actionable research targets."""),
        ("human", "User Query: {query}\n\nSession Context: {context}"),
    ])

    context = ""
    if state.get("config_overrides"):
        context = f"Config overrides: {state['config_overrides']}"

    try:
        plan = await prompt | llm.ainvoke({
            "query": state["user_query"],
            "context": context,
        })
        state["research_plan"] = plan
    except Exception as e:
        # Fallback plan
        state["research_plan"] = ResearchPlan(
            query=state["user_query"],
            vendor_focus=[],
            market_focus=["channel trends", "technology refresh cycles"],
            partner_focus=["MSPs", "VARs", "system integrators"],
            priority="medium",
        )

    return update_step(state, "vendor_research")


async def vendor_research_node(state: ChannelIntelState, config: RunnableConfig) -> ChannelIntelState:
    """Execute vendor research using AutoGen agent."""
    plan = state.get("research_plan")
    if not plan or not plan.vendor_focus:
        # Auto-extract vendor names from query if not in plan
        vendor_names = _extract_vendor_names(state["user_query"])
    else:
        vendor_names = plan.vendor_focus[:plan.max_vendors]

    if vendor_names:
        results = await run_filings_agent(vendor_names, state["session_id"])
        state["vendor_research"] = results

    return update_step(state, "market_research")


async def market_research_node(state: ChannelIntelState, config: RunnableConfig) -> ChannelIntelState:
    """Execute market research using LangGraph market agent."""
    plan = state.get("research_plan")
    if not plan:
        topics = ["channel trends", "technology refresh cycles", "partner ecosystem"]
    else:
        topics = plan.market_focus

    signals = await run_market_agent(topics, state["session_id"])
    state["market_signals"] = signals

    return update_step(state, "partner_research")


async def partner_research_node(state: ChannelIntelState, config: RunnableConfig) -> ChannelIntelState:
    """Execute partner research using CrewAI agent."""
    plan = state.get("research_plan")
    if not plan:
        partner_segments = ["MSPs", "VARs", "system integrators", "cloud providers"]
    else:
        partner_segments = plan.partner_focus[:plan.max_partners]

    profiles = await run_stakeholder_agent(partner_segments, state["session_id"])
    state["partner_profiles"] = profiles

    return update_step(state, "synthesis")


async def synthesis_node(state: ChannelIntelState, config: RunnableConfig) -> ChannelIntelState:
    """Synthesize all research into actionable briefing."""
    result = await run_synthesis(
        vendor_research=state.get("vendor_research", {}),
        market_signals=state.get("market_signals", []),
        partner_profiles=state.get("partner_profiles", {}),
        user_query=state["user_query"],
        session_id=state["session_id"],
    )

    state["briefing_draft"] = result["draft"]
    state["briefing_final"] = result["final"]
    state["opportunities"] = result["opportunities"]
    state["citations"] = result["citations"]

    return update_step(state, "complete")


async def completion_node(state: ChannelIntelState, config: RunnableConfig) -> ChannelIntelState:
    """Finalize and mark complete."""
    state["is_complete"] = True
    state["end_time"] = datetime.now()
    return state


def _extract_vendor_names(query: str) -> list[str]:
    """Extract potential vendor names from query using simple heuristics."""
    # Common vendor indicators
    vendors = []
    query_lower = query.lower()

    # Known major vendors in channel
    known_vendors = [
        "dell", "hp", "hewlett packard", "lenovo", "cisco", "microsoft", "aws", "amazon web services",
        "azure", "google cloud", "gcp", "vmware", "broadcom", "nvidia", "intel", "amd",
        "ibm", "oracle", "salesforce", "service now", "splunk", "datadog", "crowdstrike",
        "palo alto", "fortinet", "check point", "f5", "citrix", "nutanix", "pure storage",
        "netapp", "hitachi", "fujitsu", "hpe", "arista", "juniper", "extreme networks",
    ]

    for vendor in known_vendors:
        if vendor in query_lower:
            vendors.append(vendor.title())

    # Also look for capitalized words that might be vendors
    import re
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    for word in capitalized:
        if word.lower() not in [v.lower() for v in vendors] and len(word) > 2:
            vendors.append(word)

    return vendors[:5]  # Limit to 5


def should_continue(state: ChannelIntelState) -> Literal["continue", "end"]:
    """Conditional edge: continue or end based on completion."""
    if state.get("is_complete"):
        return "end"
    if state.get("error"):
        return "end"
    if state["iteration_count"] >= state["max_iterations"]:
        state["error"] = "Max iterations reached"
        return "end"
    return "continue"


def build_supervisor_graph() -> StateGraph:
    """Build the main supervisor LangGraph."""

    workflow = StateGraph(ChannelIntelState)

    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("vendor_research", vendor_research_node)
    workflow.add_node("market_research", market_research_node)
    workflow.add_node("partner_research", partner_research_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("completion", completion_node)

    # Add edges
    workflow.set_entry_point("planning")

    workflow.add_edge("planning", "vendor_research")
    workflow.add_edge("vendor_research", "market_research")
    workflow.add_edge("market_research", "partner_research")
    workflow.add_edge("partner_research", "synthesis")
    workflow.add_edge("synthesis", "completion")
    workflow.add_edge("completion", END)

    # Add conditional edge for early termination
    workflow.add_conditional_edges(
        "planning",
        should_continue,
        {"continue": "vendor_research", "end": END},
    )

    return workflow


# Compiled graph instance
_graph = None


def get_supervisor_graph() -> StateGraph:
    """Get compiled supervisor graph (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_supervisor_graph().compile(
            checkpointer=MemorySaver(),
        )
    return _graph


async def run_research(
    user_query: str,
    session_id: str,
    user_id: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ChannelIntelState:
    """
    Run the complete research pipeline.

    Args:
        user_query: Natural language research query
        session_id: Unique session identifier
        user_id: Optional user identifier
        config_overrides: Optional configuration overrides

    Returns:
        Final ChannelIntelState with all research results
    """
    graph = get_supervisor_graph()

    initial_state = create_initial_state(
        user_query=user_query,
        session_id=session_id,
        user_id=user_id,
        config_overrides=config_overrides,
    )

    config = RunnableConfig(
        configurable={"thread_id": session_id},
        recursion_limit=25,
    )

    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state