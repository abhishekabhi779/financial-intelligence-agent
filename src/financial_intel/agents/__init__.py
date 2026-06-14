"""
Channel Intelligence Agent — Agents Package
"""

from financial_intel.agents.filings_agent import (
    run_filings_agent,
    simple_filings_research,
)
from financial_intel.agents.stakeholder_agent import (
    run_stakeholder_agent,
)

__all__ = [
    "run_filings_agent",
    "simple_filings_research",
    "run_stakeholder_agent",
]