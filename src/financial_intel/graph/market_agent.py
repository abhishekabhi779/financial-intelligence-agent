"""
Channel Intelligence Agent — Market Agent (LangGraph State Machine)

Monitors market trends, competitor moves, earnings, and industry signals.
"""

from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from financial_intel.config import get_settings, get_llm_config
from financial_intel.state import (
    MarketAgentState,
    MarketSignal,
    TokenUsage,
)
from financial_intel.tools.registry import get_tool


class SearchQueries(BaseModel):
    """Structured search queries for market research."""

    queries: List[str] = Field(description="List of search queries to execute")
    reasoning: str = Field(description="Why these queries")


class SignalExtraction(BaseModel):
    """Extracted market signals from search results."""

    signals: List[MarketSignal] = Field(description="Extracted market signals")


async def generate_queries_node(state: MarketAgentState, config: RunnableConfig) -> MarketAgentState:
    """Generate search queries for each research topic."""
    from financial_intel.config import get_llm_config

    llm_config = get_llm_config("market_agent")
    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    ).with_structured_output(SearchQueries)

    topics = state.get("research_topics", [])
    current_topic = topics[state.get("current_topic_index", 0)] if topics else "channel trends"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate 3-5 specific search queries to research the given market topic.
Focus on recent developments (last 30 days), competitor moves, earnings reports,
product launches, funding rounds, and regulatory changes in the IT channel."""),
        ("human", "Research Topic: {topic}\n\nGenerate search queries for web search and news APIs."),
    ])

    result = await (prompt | llm).ainvoke({"topic": current_topic})
    state["search_queries"] = result.queries
    return state


async def execute_search_node(state: MarketAgentState, config: RunnableConfig) -> MarketAgentState:
    """Execute web searches for generated queries."""
    search_tool = get_tool("web_search")
    news_tool = get_tool("news")

    all_results = []

    for query in state.get("search_queries", []):
        # Web search
        web_results = await search_tool.ainvoke({"query": query, "max_results": 5})
        all_results.extend([
            {"source": "web", "query": query, **r} for r in web_results
        ])

        # News search
        news_results = await news_tool.ainvoke({"query": query, "max_results": 3})
        all_results.extend([
            {"source": "news", "query": query, **r} for r in news_results
        ])

    state["raw_search_results"] = all_results
    return state


async def extract_signals_node(state: MarketAgentState, config: RunnableConfig) -> MarketAgentState:
    """Extract structured market signals from search results."""
    from financial_intel.config import get_llm_config

    llm_config = get_llm_config("market_agent")
    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    ).with_structured_output(SignalExtraction)

    results = state.get("raw_search_results", [])
    if not results:
        state["signals_collected"] = state.get("signals_collected", [])
        return state

    # Format results for LLM
    formatted = "\n\n".join([
        f"Source: {r.get('source', 'unknown')}\nQuery: {r.get('query', '')}\nTitle: {r.get('title', '')}\nSnippet: {r.get('snippet', r.get('content', ''))}\nURL: {r.get('url', '')}"
        for r in results[:20]  # Limit context
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract market intelligence signals from the search results.
Identify: trends, competitor moves, earnings, product launches, funding, regulations.
For each signal, provide type, title, summary, source, relevance score (0-1), sentiment."""),
        ("human", "Search Results:\n{results}\n\nExtract structured market signals."),
    ])

    result = await (prompt | llm).ainvoke({"results": formatted})
    new_signals = result.signals

    # Deduplicate by title similarity
    existing_titles = {s.title.lower() for s in state.get("signals_collected", [])}
    for signal in new_signals:
        if signal.title.lower() not in existing_titles:
            state.setdefault("signals_collected", []).append(signal)
            existing_titles.add(signal.title.lower())

    return state


async def next_topic_node(state: MarketAgentState, config: RunnableConfig) -> MarketAgentState:
    """Move to next research topic."""
    current = state.get("current_topic_index", 0)
    topics = state.get("research_topics", [])
    state["current_topic_index"] = current + 1

    if state["current_topic_index"] >= len(topics):
        state["is_complete"] = True
    return state


def should_continue_market(state: MarketAgentState) -> Literal["continue", "extract", "next", "end"]:
    """Market agent routing logic."""
    if state.get("is_complete"):
        return "end"

    if not state.get("search_queries"):
        return "continue"  # Generate queries

    if not state.get("raw_search_results"):
        return "continue"  # Execute search

    if not state.get("signals_collected") or state["current_topic_index"] < len(state.get("research_topics", [0])) - 1:
        return "extract"  # Extract signals

    return "next"  # Move to next topic


def build_market_agent_graph() -> StateGraph:
    """Build the market agent LangGraph."""

    workflow = StateGraph(MarketAgentState)

    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("execute_search", execute_search_node)
    workflow.add_node("extract_signals", extract_signals_node)
    workflow.add_node("next_topic", next_topic_node)

    workflow.set_entry_point("generate_queries")

    workflow.add_conditional_edges(
        "generate_queries",
        should_continue_market,
        {
            "continue": "execute_search",
            "extract": "extract_signals",
            "next": "next_topic",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "execute_search",
        should_continue_market,
        {
            "continue": "execute_search",
            "extract": "extract_signals",
            "next": "next_topic",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "extract_signals",
        should_continue_market,
        {
            "continue": "generate_queries",
            "extract": "extract_signals",
            "next": "next_topic",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "next_topic",
        should_continue_market,
        {
            "continue": "generate_queries",
            "extract": "extract_signals",
            "next": "next_topic",
            "end": END,
        },
    )

    return workflow


_market_graph = None


def get_market_agent_graph() -> StateGraph:
    """Get compiled market agent graph."""
    global _market_graph
    if _market_graph is None:
        _market_graph = build_market_agent_graph().compile()
    return _market_graph


async def run_market_agent(
    topics: List[str],
    session_id: str,
) -> List[MarketSignal]:
    """
    Run market research agent for given topics.

    Args:
        topics: List of market research topics
        session_id: Session identifier for tracing

    Returns:
        List of MarketSignal objects
    """
    graph = get_market_agent_graph()

    initial_state = MarketAgentState(
        research_topics=topics,
        signals_collected=[],
        current_topic_index=0,
        search_queries=[],
        raw_search_results=[],
        is_complete=False,
        error=None,
    )

    config = RunnableConfig(configurable={"thread_id": f"{session_id}-market"})

    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state.get("signals_collected", [])