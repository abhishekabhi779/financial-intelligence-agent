"""
Channel Intelligence Agent — Partner Agent (CrewAI)

Role-based crew for partner profiling and opportunity scoring.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from financial_intel.config import get_settings, get_llm_config
from financial_intel.state import PartnerProfile
from financial_intel.tools.registry import get_tool
from financial_intel.tools.core import web_search, vector_search


class PartnerResearchOutput(BaseModel):
    """Structured output from partner research."""

    profiles: Dict[str, PartnerProfile]
    opportunities: List[Dict[str, Any]]


def create_stakeholder_agents(llm: ChatOpenAI) -> List[Agent]:
    """Create CrewAI agents for partner research."""

    web_search_tool = get_tool("web_search")
    vector_search_tool = get_tool("vector_search")

    # Researcher Agent
    researcher = Agent(
        role="Partner Research Specialist",
        goal="Gather comprehensive firmographic and technographic data on channel partners",
        backstory="""You are an expert in channel partner ecosystems with deep knowledge of
MSPs, VARs, system integrators, and cloud service providers. You know how to find
detailed company profiles, technology stacks, vendor relationships, and buying signals
from public sources, financial filings, and industry databases.""",
        verbose=True,
        allow_delegation=False,
        tools=[web_search_tool, vector_search_tool] if web_search_tool else [],
        llm=llm,
    )

    # Analyst Agent
    analyst = Agent(
        role="Partner Intelligence Analyst",
        goal="Analyze partner data to assess channel fit, engagement readiness, and opportunity potential",
        backstory="""You specialize in partner scoring models and opportunity qualification.
You evaluate partners on: technical capabilities, market reach, vendor alignments,
financial health, and buying intent signals. You identify which partners are ready
to engage with specific vendor solutions.""",
        verbose=True,
        allow_delegation=False,
        tools=[vector_search_tool] if vector_search_tool else [],
        llm=llm,
    )

    # Scorer Agent
    scorer = Agent(
        role="Channel Opportunity Scorer",
        goal="Generate prioritized opportunity scores with clear rationale for sales teams",
        backstory="""You build weighted scoring models for channel opportunities.
You consider: vendor-product fit (40%), partner readiness (35%), market timing (25%).
You output actionable recommendations with specific next steps for account executives.""",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=llm,
    )

    return [researcher, analyst, scorer]


def create_partner_tasks(agents: List[Agent], partner_segments: List[str], vendor_context: str = "") -> List[Task]:
    """Create tasks for the partner research crew."""

    researcher, analyst, scorer = agents

    # Task 1: Research partners
    research_task = Task(
        description=f"""Research the following partner segments: {', '.join(partner_segments)}

For each segment, identify 3-5 representative companies and gather:
1. Firmographics: size, revenue, geography, growth rate
2. Specializations: vertical focus, technology domains, certifications
3. Vendor partnerships: current vendor relationships, partner tier levels
4. Technology stack: cloud platforms, security, infrastructure, applications
5. Buying signals: recent funding, hiring, expansions, tech refresh cycles
6. Contact info: key decision makers, engagement history

Vendor context: {vendor_context}

Output: Structured profiles for each company with source citations.""",
        expected_output="JSON with partner profiles including all firmographic/technographic data",
        agent=researcher,
    )

    # Task 2: Analyze and score
    analysis_task = Task(
        description="""Analyze the researched partner profiles against the vendor context.

For each partner, assess:
1. Technical Fit: Does their stack/specialization align with vendor solutions?
2. Market Reach: Do they serve the vendor's target markets/verticals?
3. Vendor Alignment: Are they already partnered with complementary/non-competing vendors?
4. Readiness Signals: Budget cycles, hiring, expansions, refresh timing
5. Engagement Probability: Likelihood to respond to outreach

Score each dimension 0-1 and provide rationale.""",
        expected_output="JSON with partner scores and dimensional analysis",
        agent=analyst,
        context=[research_task],
    )

    # Task 3: Generate opportunities
    scoring_task = Task(
        description="""Generate prioritized channel opportunities from analyzed partners.

Create 3-5 high-confidence opportunities with:
1. Partner name & vendor match
2. Opportunity type: new_logo, expansion, renewal, cross_sell, tech_refresh
3. Composite score (0.4*vendor_fit + 0.35*readiness + 0.25*timing)
4. Estimated deal size range
5. Specific next actions for sales team
6. Supporting evidence with citations

Format as actionable sales briefing.""",
        expected_output="JSON with scored opportunities and next actions",
        agent=scorer,
        context=[research_task, analysis_task],
    )

    return [research_task, analysis_task, scoring_task]


async def run_stakeholder_agent(
    partner_segments: List[str],
    session_id: str,
    vendor_context: str = "",
) -> Dict[str, PartnerProfile]:
    """
    Run partner research using CrewAI.

    Args:
        partner_segments: List of partner types to research (e.g., ["MSPs", "VARs"])
        session_id: Session identifier
        vendor_context: Vendor/product context for targeted research

    Returns:
        Dict mapping partner name to PartnerProfile
    """
    settings = get_settings()
    llm_config = get_llm_config("stakeholder_agent")

    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )

    # Create agents and tasks
    agents = create_stakeholder_agents(llm)
    tasks = create_partner_tasks(agents, partner_segments, vendor_context)

    # Create crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        max_rpm=settings.agents.stakeholder_agent.get("crew", {}).get("max_rpm", 10),
    )

    # Execute
    try:
        result = await crew.kickoff_async()

        # Parse result into PartnerProfile objects
        return _parse_crew_result(result, partner_segments)

    except Exception as e:
        # Fallback to simple research
        return await _fallback_partner_research(partner_segments, session_id, vendor_context)


def _parse_crew_result(result: Any, partner_segments: List[str]) -> Dict[str, PartnerProfile]:
    """Parse CrewAI result into PartnerProfile dict."""
    profiles = {}

    # CrewAI result parsing depends on version
    # Try to extract JSON from output
    output = str(result)

    try:
        import json
        import re

        # Find JSON in output
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for partner_data in data.get("profiles", []):
                profile = PartnerProfile(**partner_data)
                profiles[profile.company_name] = profile
    except Exception:
        pass

    return profiles


async def _fallback_partner_research(
    partner_segments: List[str],
    session_id: str,
    vendor_context: str,
) -> Dict[str, PartnerProfile]:
    """
    Fallback partner research using tools directly (if CrewAI unavailable).
    """
    from financial_intel.tools.core import web_search

    profiles = {}

    # Known partner companies by segment
    partner_companies = {
        "MSPs": ["CDW", "Insight", "SHI", "Connection", "Softchoice", "NTT Data", "Accenture", "Deloitte"],
        "VARs": ["Presidio", "World Wide Technology", "CompuCom", "PCM", "Zones", "Advizex"],
        "system integrators": ["IBM Global Services", "Deloitte Consulting", "Accenture", "Capgemini", "Cognizant"],
        "cloud providers": ["AWS Partners", "Azure Experts MSPs", "Google Cloud Partners", "Rackspace", "Datapipe"],
        "security": ["Optiv", "GuidePoint Security", "Fishtech Group", "Coalfire", "Schellman"],
    }

    # Research each segment
    for segment in partner_segments:
        companies = partner_companies.get(segment.lower(), [])

        for company in companies[:3]:  # Limit to 3 per segment
            try:
                # Search for company info
                queries = [
                    f"{company} company profile revenue employees",
                    f"{company} technology partnerships certifications",
                    f"{company} specializations vertical focus",
                    f"{company} recent news funding expansion hiring",
                ]

                all_findings = []
                for query in queries:
                    results = await web_search(query, max_results=3)
                    all_findings.extend(results)

                # Build profile
                profile = _build_partner_profile(company, segment, all_findings)
                profiles[company] = profile

            except Exception:
                continue

    return profiles


def _build_partner_profile(company: str, segment: str, findings: List[Dict]) -> PartnerProfile:
    """Build PartnerProfile from search findings."""
    content = " ".join(f.get("snippet", "") for f in findings)

    # Extract info using simple heuristics
    import re

    # Revenue/employee patterns
    revenue_match = re.search(r'\$[\d.]+[BM]?\s*(?:revenue|annual)', content, re.IGNORECASE)
    employee_match = re.search(r'[\d,]+\\s*employees?', content, re.IGNORECASE)

    # Specializations
    specs = []
    spec_keywords = ["cloud", "security", "network", "data center", "AI", "analytics", "managed services",
                     "migration", "modernization", "compliance", "backup", "disaster recovery"]
    for kw in spec_keywords:
        if kw.lower() in content.lower():
            specs.append(kw.title())

    # Vendor partnerships
    vendors = []
    vendor_keywords = ["microsoft", "aws", "azure", "google cloud", "cisco", "vmware", "dell", "hp",
                       "ibm", "oracle", "salesforce", "splunk", "crowdstrike", "palo alto"]
    for v in vendor_keywords:
        if v in content.lower():
            vendors.append(v.title())

    # Buying signals
    signals = []
    signal_keywords = ["hiring", "expansion", "funding", "acquisition", "tech refresh", "modernization",
                       "migration", "new contract", "partnership announced"]
    for s in signal_keywords:
        if s in content.lower():
            signals.append(s.title())

    return PartnerProfile(
        company_name=company,
        specializations=specs[:8],
        vendor_partnerships=vendors[:10],
        buying_signals=signals[:5],
        source_urls=[f.get("url", "") for f in findings if f.get("url")][:5],
        engagement_score=0.5,
        opportunity_score=0.5,
    )