# Financial Intelligence Research Agent (FIRA)

> **Multi-agent system for automated equity research** — Built with LangGraph, AutoGen, CrewAI, and LangChain using public financial data (SEC filings, earnings calls, 13F, options flow)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![AutoGen](https://img.shields.io/badge/AutoGen-0.2+-orange.svg)](https://microsoft.github.io/autogen/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.80+-purple.svg)](https://crewai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The **Financial Intelligence Research Agent (FIRA)** is a production-ready multi-agent system that automates equity research using public financial data. It mirrors the architecture used by institutional research desks — autonomously analyzing SEC filings, earnings calls, institutional holdings, insider trades, and options flow to generate investment briefings.

**Analogous to**: Bloomberg Terminal + equity research analyst, but automated via agentic AI.

### Key Use Cases

| Use Case | Example Query | Output |
|---|---|---|
| **Earnings Intelligence** | "Summarize NVDA Q3 earnings + guidance vs peers" | Structured briefing: revenue, margins, guidance, key quotes, peer comparison |
| **SEC Filing Analysis** | "What risks did NVDA disclose in 10-K vs last year?" | Risk factor diff, red flags, material changes |
| **Peer Benchmarking** | "Compare AVGO vs MRVL on FCF conversion, R&D %, debt" | Side-by-side table with percentile rankings |
| **Catalyst Tracking** | "Upcoming catalysts for semis in Q1 2025" | Calendar: earnings, investor days, product launches, FDA decisions |
| **Institutional Flow** | "13F changes for top 10 holders of AMZN" | Buyer/seller analysis, position sizing, conviction signals |
| **Credit/Event Risk** | "Distress signals in retail sector — who's next?" | Z-scores, covenant analysis, maturity walls, short interest |

## Architecture

```
USER QUERY: "Analyze NVDA Q3 earnings vs AMD/INTC + 13F changes + option flow"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR GRAPH (LangGraph)                     │
│  ResearchPlan: tickers, form_types, lookback, data_sources, peers  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ FILINGS AGENT │    │ MARKET AGENT  │    │ STAKEHOLDER   │
│ (AutoGen)     │    │ (LangGraph)   │    │ AGENT (CrewAI)│
├───────────────┤    ├───────────────┤    ├───────────────┤
│ • Researcher  │    │ • Price/Vol   │    │ • 13F Analyst │
│   (SEC API)   │    │ • Options     │    │ • Insider     │
│ • Analyst     │    │ • Estimates   │    │ • Credit      │
│   (diff, XBRL)│    │ • Calendar    │    │               │
│ Round-robin   │    │ State machine │    │ Sequential    │
│ max 6 rounds  │    │               │    │ process       │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHESIS AGENT (LangChain LCEL)                 │
│  Score = 0.40×Fundamental + 0.25×Valuation + 0.25×Catalyst + 0.10×Sentiment
│  Output: Investment Thesis → Metrics → Risks → Catalysts → Action  │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack Alignment

| Requirement | Implementation |
|---|---|
| **LangGraph** | Supervisor graph + Market Agent state machine |
| **AutoGen** | Filings Agent (Researcher + Analyst) |
| **CrewAI** | Stakeholder Agent (13F/Insider/Credit roles) |
| **LangChain LCEL** | Synthesis pipeline with structured output |
| **Vector DB** | ChromaDB (local), Pinecone/Weaviate (prod) |
| **LLMs** | GPT-4o, Claude 3.5, Gemini 1.5 via LiteLLM |
| **Observability** | LangSmith + Prometheus + Cost tracking |
| **CI/CD** | Ruff, MyPy, pytest, Bandit, Docker |

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- API Keys: OpenAI, Anthropic, Google, SEC API, Alpha Vantage, NewsAPI, LangSmith

### Installation

```bash
# Clone and navigate
git clone https://github.com/abhishekabhi779/financial-intelligence-agent.git
cd financial-intelligence-agent

# Install dependencies
make install-dev

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start development server
make dev
```

### API Keys Required (Free Tiers Available)

```env
# SEC Filings (choose one)
SEC_API_KEY=***              # sec-api.io (100 free/mo)
# OR use edgartools (no key needed)

# Earnings & Market Data
ALPHA_VANTAGE_API_KEY=***    # 25 req/day free
TWELVE_DATA_API_KEY=***      # 800 req/day free

# News & Sentiment
NEWSAPI_KEY=***              # 100 req/day free

# Observability
LANGSMITH_API_KEY=***        # Tracing

# LLM Providers
OPENAI_API_KEY=***
ANTHROPIC_API_KEY=***
GOOGLE_API_KEY=***
```

## Usage

### CLI

```bash
# Full research briefing
make run QUERY="NVDA Q3 earnings vs AMD/INTC + 13F changes"

# Or use CLI directly
poetry run fi research "AVGO vs MRVL peer comparison" --output briefing.json

# SEC filings deep-dive
poetry run fi filings "NVDA" --forms 10-K,10-Q --output filings.json

# Institutional/insider analysis
poetry run fi stakeholder "NVDA" --types 13f,insider --output holders.json
```

### REST API

```bash
# Start server
make dev  # or: poetry run uvicorn financial_intel.api.main:app --reload

# Research endpoint
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "NVDA Q3 earnings analysis with peer comparison"}'

# Streaming endpoint (SSE)
curl -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Distress signals in retail sector"}'

# Health check
curl http://localhost:8000/health

# Metrics (Prometheus)
curl http://localhost:8000/metrics
```

### Python API

```python
import asyncio
from financial_intel.graph.supervisor import run_research

async def main():
    result = await run_research(
        user_query="Analyze NVDA Q3 earnings vs AMD/INTC with 13F changes",
        session_id="my-session-001",
    )
    
    print(result["briefing_final"])
    for opp in result["opportunities"]:
        print(f"🎯 {opp.ticker} — Score: {opp.score:.0%}")

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
    filings_agent: gpt-4o
    market_agent: gpt-4o-mini
    stakeholder_agent: gpt-4o
    synthesis: gpt-4o

scoring:
  fundamental_quality:
    weights:
      roic_percentile: 0.25
      fcf_margin_percentile: 0.20
      revenue_growth_percentile: 0.20
      gross_margin_percentile: 0.15
      debt_to_ebitda_inverse: 0.10
      piotroski_f_score: 0.10
  
  valuation:
    weights:
      pe_percentile_inverse: 0.30
      ev_ebitda_percentile_inverse: 0.25
      fcf_yield_percentile: 0.20
      peg_percentile_inverse: 0.15
      price_sales_percentile_inverse: 0.10
  
  catalyst:
    weights:
      earnings_momentum: 0.35
      guidance_revision: 0.25
      institutional_accumulation: 0.20
      insider_buying: 0.10
      option_flow_bullish: 0.10

  composite:
    fundamental_quality: 0.40
    valuation: 0.25
    catalyst: 0.25
    sentiment: 0.10

agents:
  filings_agent:
    max_rounds: 6
  stakeholder_agent:
    crew:
      max_rpm: 10

observability:
  langsmith:
    enabled: true
    project_name: financial-intel
  cost_tracking:
    budget_usd_per_day: 50.0
```

## Project Structure

```
financial-intelligence-agent/
├── AGENTS.md                          # Instructions for AI agents
├── configs/
│   ├── settings.yaml                  # All config (LLM, RAG, agents, scoring)
│   ├── prompts/                       # Agent prompts (YAML)
│   └── eval/
│       └── golden_sets.jsonl          # 20+ evaluation benchmarks
├── src/financial_intel/
│   ├── config.py                      # Pydantic Settings loader
│   ├── state.py                       # TypedDict state + Financial Pydantic models
│   ├── graph/
│   │   ├── supervisor.py              # Main LangGraph supervisor
│   │   ├── market_agent.py            # Market data state machine
│   │   └── synthesis.py               # LCEL synthesis pipeline
│   ├── agents/
│   │   ├── filings_agent.py           # AutoGen: SEC filings analysis
│   │   └── stakeholder_agent.py       # CrewAI: 13F/Insider/Credit
│   ├── tools/
│   │   ├── registry.py                # MCP-style tool registry
│   │   ├── core.py                    # Web search, news, vector search
│   │   └── finance.py                 # 🆕 SEC, earnings, 13F, options, calendar
│   ├── rag/
│   │   └── pipeline.py                # Ingestion + hybrid retrieval
│   ├── api/
│   │   └── main.py                    # FastAPI application
│   ├── observability/                 # LangSmith, Prometheus, cost tracking
│   └── cli/                           # Typer CLI (command: `fi`)
├── tests/
├── docker/
├── pyproject.toml
├── Makefile
├── docker-compose.yml
└── README.md
```

## Data Sources & Tools

| Tool | Category | Source | Free Tier |
|---|---|---|---|
| `sec_filing_search` | sec | sec-api.io / edgartools | 100/mo / unlimited |
| `sec_xbrl_extract` | sec | SEC XBRL | free |
| `sec_filing_diff` | sec | Custom diff engine | free |
| `earnings_transcript` | earnings | Alpha Vantage | 25/day |
| `fundamental_snapshot` | market | Alpha Vantage / yfinance | 25/day / unlimited |
| `thirteen_f_changes` | institutional | sec-api.io | 100/mo |
| `insider_trades` | insider | sec-api.io / OpenInsider | 100/mo / free |
| `unusual_options_flow` | options | CBOE / ThetaData | paid |
| `events_calendar` | calendar | Multiple | free |

## Evaluation

Run evaluation benchmarks:

```bash
# Run quality evaluation
poetry run pytest tests/eval/ -v

# View eval results
open htmlcov/index.html
```

Benchmarks include (`configs/eval/golden_sets.jsonl`):
- Earnings analysis accuracy (vs. golden Q&A)
- SEC filing diff detection
- Peer benchmarking table correctness
- 13F change detection + conviction scoring
- Distress dashboard signal quality
- Hallucination rate < 20%
- Cost per research session tracking

## Deployment

### Production Docker

```bash
# Build
make build

# Run
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=*** \
  -e ANTHROPIC_API_KEY=*** \
  -e SEC_API_KEY=*** \
  -e ALPHA_VANTAGE_API_KEY=*** \
  -e LANGSMITH_API_KEY=*** \
  financial-intel:latest
```

### Cloud Options

| Platform | Command |
|---|---|
| **GCP Cloud Run** | `gcloud run deploy --image=financial-intel` |
| **AWS ECS/Fargate** | `ecs-cli compose up` |
| **Azure Container Apps** | `az containerapp up` |
| **Kubernetes** | `kubectl apply -f k8s/` |

## Roadmap (Where to Improve)

### High Priority
1. **SEC Filing Diff Engine** — Automated 10-K/10-Q section diffing with new risk factor highlights
2. **Guidance Tracker** — Parse every earnings call for guidance changes vs consensus, track accuracy
3. **13F Conviction Scoring** — Weight holders by track record, holding period, position sizing
4. **Options Flow + Filings Fusion** — Correlate unusual put/call activity with 8-K events
5. **Congressional Trading Overlay** — Quiver Quantitative senator trades vs sector exposure

### Medium Priority
6. **Distress Dashboard** — Automated Z-score, covenant monitoring, maturity wall visualization
7. **Earnings Prep Packs** — Auto-generated 2-pagers before every earnings call
8. **Real-time Alerts** — 8-K triggers, 13F changes, option blocks → webhook/Slack
9. **Factor Model Integration** — Barra-style risk factors, style attribution
10. **Backtesting Framework** — Score → forward returns, IC decay analysis

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

- **SEC** for EDGAR public filings access
- **Alpha Vantage** for free earnings call transcripts
- **LangChain/LangGraph** team for orchestration framework
- **Microsoft AutoGen** team for multi-agent conversation
- **CrewAI** team for role-based agent crews
- **ChromaDB** for vector storage

---

**Built for the Associate Agentic AI Engineer portfolio** — Demonstrating hands-on experience with LangGraph, AutoGen, CrewAI, LangChain for financial intelligence.