"""
Channel Intelligence Agent — Graph Package
"""

from financial_intel.graph.supervisor import (
    build_supervisor_graph,
    get_supervisor_graph,
    run_research,
)
from financial_intel.graph.market_agent import (
    build_market_agent_graph,
    get_market_agent_graph,
    run_market_agent,
)
from financial_intel.graph.synthesis import (
    create_synthesis_chain,
    get_synthesis_chain,
    run_synthesis,
)

__all__ = [
    "build_supervisor_graph",
    "get_supervisor_graph",
    "run_research",
    "build_market_agent_graph",
    "get_market_agent_graph",
    "run_market_agent",
    "create_synthesis_chain",
    "get_synthesis_chain",
    "run_synthesis",
]