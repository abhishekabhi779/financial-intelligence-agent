"""
Channel Intelligence Agent — Vendor Agent (AutoGen)

Multi-agent conversation for deep vendor research using AutoGen.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from financial_intel.config import get_settings, get_llm_config
from financial_intel.state import VendorInfo
from financial_intel.tools.registry import get_tool
from financial_intel.tools.core import web_search, vendor_api


class VendorResearchResult(BaseModel):
    """Result from vendor research agent."""

    vendor_name: str
    info: VendorInfo
    conversation_log: List[Dict[str, Any]]


async def create_filings_agents(vendor_name: str, llm_client: ChatCompletionClient):
    """Create AutoGen agents for vendor research."""

    # Researcher Agent - gathers raw information
    researcher = AssistantAgent(
        name="vendor_researcher",
        model_client=llm_client,
        system_message=f"""You are a vendor research specialist for {vendor_name}.
Your role: Gather comprehensive, accurate information about {vendor_name}.

Focus areas:
1. Product portfolio & categories
2. Pricing models & strategies
3. Target markets & customer segments
4. Partner program structure & benefits
5. Recent product launches & roadmap
6. Competitive positioning
7. Financial performance (public info only)

Use tools to search for current information. Cite sources with URLs.
Be specific - mention actual product names, price ranges, program tiers.
Output findings as structured JSON.""",
        tools=[web_search, vendor_api],
    )

    # Analyst Agent - synthesizes and validates
    analyst = AssistantAgent(
        name="vendor_analyst",
        model_client=llm_client,
        system_message=f"""You are a vendor analyst for {vendor_name}.
Your role: Analyze and validate research findings, identify gaps, provide strategic insights.

Tasks:
1. Cross-reference researcher findings for consistency
2. Identify missing critical information
3. Assess channel fit for {vendor_name}
4. Highlight competitive advantages/disadvantages
5. Note any red flags or concerns
6. Provide confidence scores for each finding

Output structured analysis as JSON.""",
        tools=[web_search],
    )

    # User Proxy to manage the conversation
    user_proxy = UserProxyAgent(
        name="research_coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
    )

    return researcher, analyst, user_proxy


async def run_filings_agent(
    vendor_names: List[str],
    session_id: str,
) -> Dict[str, VendorInfo]:
    """
    Run vendor research for multiple vendors using AutoGen.

    Args:
        vendor_names: List of vendor names to research
        session_id: Session identifier for tracing

    Returns:
        Dict mapping vendor name to VendorInfo
    """
    settings = get_settings()
    llm_config = get_llm_config("filings_agent")

    llm_client = OpenAIChatCompletionClient(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )

    results = {}
    max_rounds = settings.agents.filings_agent.get("max_rounds", 8)

    for vendor_name in vendor_names:
        try:
            vendor_info = await _research_single_vendor(
                vendor_name, llm_client, max_rounds, session_id
            )
            results[vendor_name] = vendor_info
        except Exception as e:
            # Create minimal info on failure
            results[vendor_name] = VendorInfo(
                name=vendor_name,
                description=f"Research failed: {str(e)}",
                confidence_score=0.0,
            )

    return results


async def _research_single_vendor(
    vendor_name: str,
    llm_client: ChatCompletionClient,
    max_rounds: int,
    session_id: str,
) -> VendorInfo:
    """Research a single vendor using AutoGen group chat."""

    researcher, analyst, coordinator = await create_filings_agents(vendor_name, llm_client)

    # Create group chat with termination condition
    termination = MaxMessageTermination(max_messages=max_rounds)
    team = RoundRobinGroupChat(
        participants=[researcher, analyst],
        termination_condition=termination,
    )

    # Initial task
    task = f"""Research {vendor_name} comprehensively for channel sales intelligence.

Required output (JSON):
{{
  "name": "{vendor_name}",
  "description": "Company overview",
  "product_categories": ["category1", "category2"],
  "pricing_model": "subscription/tiered/perpetual/etc",
  "target_markets": ["market1", "market2"],
  "partner_program": "Program name and key details",
  "roadmap_highlights": ["upcoming1", "upcoming2"],
  "competitive_positioning": "Brief competitive analysis",
  "source_urls": ["url1", "url2"],
}}

Researcher: Start by gathering current product catalog, pricing, partner program info.
Analyst: Validate findings, add strategic insights, identify gaps.

Both: Use tools to search for real-time information. Cite all sources."""

    # Run the conversation
    result = await team.run(task=task)

    # Parse final output from conversation
    vendor_info = _parse_vendor_output(result, vendor_name)
    return vendor_info


def _parse_vendor_output(result: Any, vendor_name: str) -> VendorInfo:
    """Parse AutoGen conversation result into VendorInfo."""
    # Get the last message which should contain the structured output
    messages = result.messages if hasattr(result, 'messages') else []

    # Look for JSON in messages
    import re
    for msg in reversed(messages):
        content = getattr(msg, 'content', str(msg))
        if isinstance(content, str):
            # Try to find JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return VendorInfo(**data)
                except Exception:
                    continue

    # Fallback: extract from conversation
    return _extract_from_conversation(messages, vendor_name)


def _extract_from_conversation(messages: List[Any], vendor_name: str) -> VendorInfo:
    """Extract vendor info from conversation messages."""
    all_content = " ".join(str(getattr(m, 'content', m)) for m in messages)

    # Simple extraction heuristics
    import re

    # Extract product categories
    categories = []
    cat_patterns = [
        r'(?:categories|products?)\s*[:\-]\s*([^.]+)',
        r'(?:servers?|storage|networking|security|cloud|pcs?|laptops?|workstations?)',
    ]
    for pattern in cat_patterns:
        matches = re.findall(pattern, all_content, re.IGNORECASE)
        categories.extend([m.strip() for m in matches if isinstance(m, str)])

    # Extract URLs
    urls = re.findall(r'https?://[^\s]+', all_content)

    return VendorInfo(
        name=vendor_name,
        description=all_content[:500] if all_content else f"Research completed for {vendor_name}",
        product_categories=list(set(categories))[:10],
        source_urls=list(set(urls))[:10],
        confidence_score=0.7,
    )


# Alternative: Simple non-AutoGen vendor researcher for fallback
async def simple_filings_research(vendor_names: List[str], session_id: str) -> Dict[str, VendorInfo]:
    """
    Simple vendor research using tools directly (fallback if AutoGen unavailable).
    """
    from financial_intel.tools.core import web_search, vendor_api

    results = {}

    for vendor_name in vendor_names:
        # Search for vendor info
        queries = [
            f"{vendor_name} product catalog portfolio",
            f"{vendor_name} partner program tiers benefits",
            f"{vendor_name} pricing model",
            f"{vendor_name} recent product launch roadmap",
        ]

        all_findings = []
        for query in queries:
            search_results = await web_search(query, max_results=5)
            all_findings.extend(search_results)

        # Get vendor API data
        api_data = await vendor_api(vendor_name, "catalog")
        partner_data = await vendor_api(vendor_name, "partners")

        # Synthesize
        vendor_info = VendorInfo(
            name=vendor_name,
            description=f"Research findings for {vendor_name}",
            product_categories=_extract_categories(all_findings, api_data),
            pricing_model=_extract_pricing(all_findings, api_data),
            partner_program=_extract_partner_program(partner_data),
            source_urls=[r.get("url", "") for r in all_findings if r.get("url")],
            confidence_score=0.6,
        )

        results[vendor_name] = vendor_info

    return results


def _extract_categories(findings: List[Dict], api_data: Dict) -> List[str]:
    categories = set()
    for f in findings:
        content = f.get("snippet", "").lower()
        for cat in ["server", "storage", "network", "security", "cloud", "laptop", "desktop", "workstation"]:
            if cat in content:
                categories.add(cat.title())
    if api_data.get("categories"):
        categories.update(api_data["categories"])
    return list(categories)[:10]


def _extract_pricing(findings: List[Dict], api_data: Dict) -> str:
    if api_data.get("pricing", {}).get("model"):
        return api_data["pricing"]["model"]
    for f in findings:
        if "pricing" in f.get("snippet", "").lower():
            return f.get("snippet", "")[:200]
    return "Not specified"


def _extract_partner_program(partner_data: Dict) -> str:
    if partner_data.get("program"):
        return partner_data["program"]
    if partner_data.get("tiers"):
        return f"Tiers: {', '.join(partner_data['tiers'])}"
    return "Information not available"