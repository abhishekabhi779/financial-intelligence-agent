"""
Channel Intelligence Agent — Synthesis Agent (LangChain LCEL)

Cross-references all research, scores opportunities, generates briefings.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from financial_intel.config import get_llm_config
from financial_intel.state import (
    VendorInfo,
    MarketSignal,
    PartnerProfile,
    Opportunity,
    TokenUsage,
)
from financial_intel.tools.registry import get_tool


class OpportunityScore(BaseModel):
    """Scored opportunity with rationale."""

    id: str
    vendor: str
    partner: str
    opportunity_type: str
    score: float
    rationale: str
    vendor_fit: float
    partner_readiness: float
    market_timing: float
    estimated_value: Optional[float] = None
    next_actions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    """Complete synthesis output."""

    draft_briefing: str
    final_briefing: str
    opportunities: List[OpportunityScore]
    citations: List[Dict[str, Any]]


def create_synthesis_chain():
    """Create the main synthesis LCEL chain."""

    llm_config = get_llm_config("synthesis")
    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )

    # Opportunity scoring chain
    scoring_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a channel sales strategist. Score opportunities based on:
1. Vendor Fit (0-1): How well does the vendor's portfolio match partner's needs?
2. Partner Readiness (0-1): Budget, timeline, technical fit, buying signals
3. Market Timing (0-1): Market trends, competitive pressure, refresh cycles

Score = 0.4*Vendor_Fit + 0.35*Partner_Readiness + 0.25*Market_Timing

Output JSON with: id, vendor, partner, opportunity_type, score, rationale,
vendor_fit, partner_readiness, market_timing, estimated_value, next_actions, supporting_evidence"""),
        ("human", """Vendor Research:
{vendor_research}

Market Signals:
{market_signals}

Partner Profiles:
{partner_profiles}

User Query: {user_query}

Generate 3-5 high-confidence opportunities. Focus on actionable, specific recommendations."""),
    ])

    scoring_chain = scoring_prompt | llm.with_structured_output(List[OpportunityScore])

    # Briefing generation chain
    briefing_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior channel analyst writing an executive briefing.
Structure:
1. Executive Summary (3-4 bullets)
2. Key Market Trends
3. Vendor Landscape
4. Partner Opportunities (table with scores)
5. Recommended Actions (prioritized)
6. Risks & Considerations

Be specific, cite sources with [n] format, keep it actionable for sales teams."""),
        ("human", "Research Data:
{vendor_research}

Market Signals:
{market_signals}

Partner Profiles:
{partner_profiles}

Scored Opportunities:
{opportunities}

User Query: {user_query}

Write a comprehensive executive briefing."""),
    ])

    briefing_chain = briefing_prompt | llm | StrOutputParser()

    # Final polish chain
    polish_prompt = ChatPromptTemplate.from_messages([
        ("system", """Polish the briefing for executive consumption.
- Tighten language, remove redundancy
- Ensure all claims have citations
- Format opportunities as clean markdown table
- Add confidence indicators
- Keep under 2000 words"""),
        ("human", "Briefing Draft:\n{draft}\n\nCitations:\n{citations}"),
    ])

    polish_chain = polish_prompt | llm | StrOutputParser()

    # Full pipeline
    chain = (
        RunnablePassthrough.assign(
            opportunities=scoring_chain,
        )
        .assign(
            draft_briefing=briefing_chain,
        )
        .assign(
            final_briefing=lambda x: polish_chain.invoke({
                "draft": x["draft_briefing"],
                "citations": x.get("citations", []),
            }),
        )
        .assign(
            citations=lambda x: _extract_citations(
                x.get("vendor_research", {}),
                x.get("market_signals", []),
                x.get("partner_profiles", {}),
            ),
        )
    )

    return chain


def _extract_citations(
    vendor_research: Dict[str, VendorInfo],
    market_signals: List[MarketSignal],
    partner_profiles: Dict[str, PartnerProfile],
) -> List[Dict[str, Any]]:
    """Extract all source citations from research."""
    citations = []
    cite_id = 1

    for vendor_name, vendor in vendor_research.items():
        for url in vendor.source_urls:
            citations.append({
                "id": cite_id,
                "type": "vendor",
                "source": vendor_name,
                "url": url,
                "title": f"{vendor_name} - {vendor.product_categories[0] if vendor.product_categories else 'Overview'}",
            })
            cite_id += 1

    for signal in market_signals:
        citations.append({
            "id": cite_id,
            "type": "market",
            "source": signal.source,
            "url": signal.source_url,
            "title": signal.title,
        })
        cite_id += 1

    for partner_name, partner in partner_profiles.items():
        for url in partner.source_urls:
            citations.append({
                "id": cite_id,
                "type": "partner",
                "source": partner_name,
                "url": url,
                "title": f"{partner_name} Profile",
            })
            cite_id += 1

    return citations


_synthesis_chain = None


def get_synthesis_chain():
    """Get synthesis chain (singleton)."""
    global _synthesis_chain
    if _synthesis_chain is None:
        _synthesis_chain = create_synthesis_chain()
    return _synthesis_chain


async def run_synthesis(
    vendor_research: Dict[str, VendorInfo],
    market_signals: List[MarketSignal],
    partner_profiles: Dict[str, PartnerProfile],
    user_query: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    Run synthesis pipeline.

    Args:
        vendor_research: Vendor research results
        market_signals: Market intelligence signals
        partner_profiles: Partner profiles
        user_query: Original user query
        session_id: Session identifier

    Returns:
        Dict with draft, final, opportunities, citations
    """
    # Convert Pydantic models to dicts for chain
    vendor_dict = {
        name: {
            "name": v.name,
            "description": v.description,
            "product_categories": v.product_categories,
            "pricing_model": v.pricing_model,
            "target_markets": v.target_markets,
            "partner_program": v.partner_program,
            "roadmap_highlights": v.roadmap_highlights,
            "competitive_positioning": v.competitive_positioning,
            "source_urls": v.source_urls,
        }
        for name, v in vendor_research.items()
    }

    market_list = [
        {
            "signal_type": s.signal_type,
            "title": s.title,
            "summary": s.summary,
            "source": s.source,
            "source_url": s.source_url,
            "relevance_score": s.relevance_score,
            "sentiment": s.sentiment,
            "entities": s.entities,
        }
        for s in market_signals
    ]

    partner_dict = {
        name: {
            "company_name": p.company_name,
            "website": p.website,
            "headquarters": p.headquarters,
            "employee_count": p.employee_count,
            "revenue_range": p.revenue_range,
            "specializations": p.specializations,
            "certifications": p.certifications,
            "vendor_partnerships": p.vendor_partnerships,
            "tech_stack": p.tech_stack,
            "buying_signals": p.buying_signals,
            "engagement_score": p.engagement_score,
            "opportunity_score": p.opportunity_score,
            "source_urls": p.source_urls,
        }
        for name, p in partner_profiles.items()
    }

    chain = get_synthesis_chain()

    result = await chain.ainvoke({
        "vendor_research": vendor_dict,
        "market_signals": market_list,
        "partner_profiles": partner_dict,
        "user_query": user_query,
    })

    # Convert opportunities back to Pydantic
    opportunities = [
        Opportunity(**opp.model_dump()) for opp in result["opportunities"]
    ]

    return {
        "draft": result["draft_briefing"],
        "final": result["final_briefing"],
        "opportunities": opportunities,
        "citations": result["citations"],
    }