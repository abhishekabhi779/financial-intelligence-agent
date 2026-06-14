# Channel Intelligence Research Agent (CIRA)

> **Multi-agent system for channel sales intelligence** — Built with LangGraph, AutoGen, CrewAI, and LangChain (Ingram Micro stack)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![AutoGen](https://img.shields.io/badge/AutoGen-0.2+-orange.svg)](https://microsoft.github.io/autogen/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.80+-purple.svg)](https://crewai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The **Channel Intelligence Research Agent (CIRA)** is a production-ready multi-agent system that mirrors Ingram Micro's agentic AI architecture. It autonomously researches vendors, market trends, and channel partners to generate actionable sales briefings — similar to Ingram Micro's **Sales Briefing Assistant** built on their **Xvantage AI Factory**.

### Key Features

| Capability | Technology | Ingram Micro Alignment |
|------------|------------|------------------------|
| **Orchestration** | LangGraph supervisor graph | AI Factory operational model |
| **Vendor Research** | AutoGen multi-agent conversation | ADK, LangChain, AutoGen, CrewAI |
| **Market Intelligence** | LangGraph state machine | 400+ internal models + external LLMs |
| **Partner Profiling** | CrewAI role-based crews | Multi-system integration (CRM, ERP) |
| **Synthesis & Briefing** | LangChain LCEL | Structured output with citations |
| **RAG Pipeline** | ChromaDB + Hybrid Search + Rerank | Vector DBs, Knowledge Graphs |
| **Observability** | LangSmith + Prometheus + Cost Tracking | MLOps/LLMOps for agent lifecycle |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SUPERVISOR (LangGraph)                          │
│  • Plans research strategy                                       │
│  • Routes to specialist agents                                   │
│  • Manages token/cost budgets                                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ VENDOR AGENT  │    │ MARKET AGENT  │    │ PARTNER AGENT │
│ (AutoGen)     │    │ (LangGraph)   │    │ (CrewAI)      │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ • Product cat │    │ • News/API    │    │ • Firmographics│
│ • Pricing     │    │ • Trends      │    │ • Tech stack  │
│ • Roadmaps    │    │ • Competitors │    │ • Buying hist │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SYNTHESIS (LangChain LCEL)                      │
│  • Cross-references findings                                     │
│  • Scores opportunities                                          │
│  • Generates structured briefing                                 │
│  • Cites sources (RAG)                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- API Keys: OpenAI, Anthropic, Google, Tavily/SerpAPI, NewsAPI, LangSmith

### Installation

```bash
# Clone and navigate
git clone <repo-url>
cd channel-intelligence-agent

# Install dependencies
make install-dev

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start development server
make dev
```

### Docker Development

```bash
# Start all services (API, ChromaDB, Jupyter, Prometheus, Grafana)
make up

# View logs
make logs

# Stop services
make down
```

## Usage

### CLI

```bash
# Run research query
make run QUERY="Which vendors should I pitch for Q1 tech refresh?"

# Or use CLI directly
poetry run channel-intel research "What are the top MSP opportunities for Dell?"

# Vendor-specific research
poetry run channel-intel vendor "Dell, Cisco, Microsoft" --output vendors.json

# Partner research
poetry run channel-intel partner "MSPs, VARs" --vendor "Dell" --output partners.json

# Ingest data for RAG
poetry run channel-intel ingest vendor "Dell" --path ./data/vendor_docs/dell_catalog.pdf
poetry run channel-intel ingest partner "CDW" --data '{"revenue": "$20B", "specializations": ["cloud", "security"]}'
```

### REST API

```bash
# Start server
make dev  # or: poetry run uvicorn channel_intel.api.main:app --reload

# Research endpoint
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Which vendors have the best partner programs for MSPs?"}'

# Streaming endpoint
curl -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Market trends for cybersecurity in 2025"}'

# Health check
curl http://localhost:8000/health

# Metrics (Prometheus)
curl http://localhost:8000/metrics
```

### Python API

```python
import asyncio
from channel_intel.graph.supervisor import run_research

async def main():
    result = await run_research(
        user_query="Which vendors should I target for H1 2025 tech refresh?",
        session_id="my-session-001",
    )
    
    print(result["briefing_final"])
    for opp in result["opportunities"]:
        print(f"- {opp.vendor} × {opp.partner}: {opp.score:.0%}")

asyncio.run(main())
```

## Configuration

All settings in `configs/settings.yaml`:

```yaml
app:
  environment: development
  log_level: DEBUG

llm:
  default_model: gpt-4o
  agent_models:
    supervisor: gpt-4o
    vendor_agent: gpt-4o
    market_agent: gpt-4o-mini
    partner_agent: gpt-4o
    synthesis: gpt-4o

rag:
  chunking_strategy: semantic
  top_k: 10
  rerank_top_k: 5
  hybrid_search_alpha: 0.5

agents:
  vendor_agent:
    max_rounds: 8
  partner_agent:
    crew:
      max_rpm: 10

observability:
  langsmith:
    enabled: true
    project_name: channel-intel
  cost_tracking:
    budget_usd_per_day: 50.0
```

## Project Structure

```
channel-intelligence-agent/
├── configs/
│   ├── settings.yaml          # Main configuration
│   ├── prompts/               # Agent prompts
│   └── eval/                  # Golden eval sets
├── src/channel_intel/
│   ├── config.py              # Pydantic Settings
│   ├── state.py               # LangGraph TypedDict state
│   ├── graph/
│   │   ├── supervisor.py      # Main orchestration graph
│   │   ├── market_agent.py    # LangGraph market agent
│   │   └── synthesis.py       # LangChain LCEL synthesis
│   ├── agents/
│   │   ├── vendor_agent.py    # AutoGen vendor research
│   │   └── partner_agent.py   # CrewAI partner profiling
│   ├── tools/
│   │   ├── registry.py        # MCP-style tool registry
│   │   └── core.py            # Web search, news, vector, vendor API
│   ├── rag/
│   │   └── pipeline.py        # Ingestion + hybrid retrieval + rerank
│   ├── api/
│   │   └── main.py            # FastAPI application
│   ├── observability/         # LangSmith, Prometheus, logging
│   └── cli/                   # Typer CLI
├── tests/
├── notebooks/
├── docker/
├── pyproject.toml
├── Makefile
└── docker-compose.yml
```

## Ingram Micro Stack Alignment

| JD Requirement | Implementation |
|----------------|----------------|
| **LangChain** | Core chains, RAG, LCEL, tool definitions |
| **LangGraph** | Supervisor graph, Market Agent state machine |
| **AutoGen** | Vendor Agent multi-agent conversation |
| **CrewAI** | Partner Agent role-based crew |
| **Google ADK** | Exploration branch (planned) |
| **Python** | Primary language throughout |
| **scikit-learn/Pandas/NumPy** | Partner scoring, trend analysis |
| **Vector DBs** | ChromaDB (local), Pinecone/Weaviate (prod) |
| **LLMs** | GPT-4o, Claude 3.5, Gemini 1.5 via LiteLLM |
| **Cloud** | Docker → Cloud Run/ECS/Container Apps |
| **Git/GitHub Actions** | CI: lint, type-check, test, build, scan |
| **Docker** | Multi-stage builds, dev/prod parity |
| **MLOps** | LangSmith tracing, cost tracking, Prometheus |

## Evaluation

Run evaluation benchmarks:

```bash
# Run quality evaluation
poetry run pytest tests/eval/ -v

# View eval results
open htmlcov/index.html
```

Benchmarks include:
- Vendor research accuracy (vs. golden Q&A)
- Market signal relevance
- Partner profiling completeness
- End-to-end briefing quality
- Hallucination rate
- Cost per research session

## Deployment

### Production Docker

```bash
# Build
make build

# Run
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  -e TAVILY_API_KEY=... \
  -e LANGSMITH_API_KEY=... \
  channel-intel:latest
```

### Cloud Options

| Platform | Command |
|----------|---------|
| **GCP Cloud Run** | `gcloud run deploy --image=channel-intel` |
| **AWS ECS/Fargate** | `ecs-cli compose up` |
| **Azure Container Apps** | `az containerapp up` |
| **Kubernetes** | `kubectl apply -f k8s/` |

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run quality checks: `make quality`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push branch: `git push origin feature/amazing-feature`
6. Open Pull Request

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- **Ingram Micro** for pioneering the Xvantage AI Factory and agentic AI architecture
- **LangChain/LangGraph** team for the orchestration framework
- **Microsoft AutoGen** team for multi-agent conversation
- **CrewAI** team for role-based agent crews
- **ChromaDB** for vector storage

---

**Built for the Associate Agentic AI Engineer role at Ingram Micro** — demonstrating hands-on experience with their exact technology stack.