"""
Channel Intelligence Agent — Core Tools

Web search, news, vector search, and vendor API tools.
"""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from financial_intel.config import get_settings
from financial_intel.tools.registry import get_registry, register_tool


# ============================================================================
# Web Search Tool
# ============================================================================

class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum results")
    search_depth: str = Field(default="advanced", description="basic or advanced")


@register_tool(
    category="search",
    name="web_search",
    description="Search the web for current information using Tavily or SerpAPI",
    tags=["web", "search", "real-time"],
)
async def web_search(query: str, max_results: int = 10, search_depth: str = "advanced") -> List[Dict[str, Any]]:
    """Search the web for current information."""
    settings = get_settings()
    provider = settings.tools.web_search.get("provider", "tavily")

    if provider == "tavily":
        return await _tavily_search(query, max_results, search_depth)
    elif provider == "serpapi":
        return await _serpapi_search(query, max_results)
    else:
        raise ValueError(f"Unknown search provider: {provider}")


async def _tavily_search(query: str, max_results: int, search_depth: str) -> List[Dict[str, Any]]:
    """Search using Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY") or get_settings().tools.web_search.get("tavily", {}).get("api_key")
    if not api_key:
        return [{"error": "Tavily API key not configured"}]

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "score": r.get("score", 0),
            "source": "tavily",
        })

    if data.get("answer"):
        results.insert(0, {
            "title": "Tavily Answer",
            "url": "",
            "snippet": data["answer"],
            "score": 1.0,
            "source": "tavily_answer",
        })

    return results


async def _serpapi_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search using SerpAPI."""
    api_key = os.getenv("SERPAPI_API_KEY") or get_settings().tools.web_search.get("serpapi", {}).get("api_key")
    if not api_key:
        return [{"error": "SerpAPI key not configured"}]

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://serpapi.com/search",
            params={
                "api_key": api_key,
                "q": query,
                "num": max_results,
                "engine": "google",
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for r in data.get("organic_results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
            "position": r.get("position", 0),
            "source": "serpapi",
        })

    return results


# ============================================================================
# News Search Tool
# ============================================================================

class NewsSearchArgs(BaseModel):
    query: str = Field(description="News search query")
    max_articles: int = Field(default=20, description="Maximum articles")
    days_back: int = Field(default=30, description="Days back to search")


@register_tool(
    category="search",
    name="news",
    description="Search news articles using NewsAPI",
    tags=["news", "real-time", "market-intelligence"],
)
async def news_search(query: str, max_articles: int = 20, days_back: int = 30) -> List[Dict[str, Any]]:
    """Search news articles."""
    settings = get_settings()
    api_key = os.getenv("NEWSAPI_KEY") or settings.tools.news.get("api_key")

    if not api_key:
        return [{"error": "NewsAPI key not configured"}]

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "apiKey": api_key,
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "language": "en",
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for article in data.get("articles", []):
        results.append({
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "snippet": article.get("description", "") or article.get("content", ""),
            "source": article.get("source", {}).get("name", ""),
            "published_at": article.get("publishedAt", ""),
            "author": article.get("author", ""),
        })

    return results


# ============================================================================
# Vector Search Tool
# ============================================================================

class VectorSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    top_k: int = Field(default=10, description="Number of results")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


@register_tool(
    category="retrieval",
    name="vector_search",
    description="Search vector database for relevant documents",
    tags=["rag", "vector", "semantic-search"],
)
async def vector_search(query: str, top_k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search vector database (ChromaDB)."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        settings = get_settings()
        persist_dir = settings.paths.chromadb
        collection_name = settings.vector_db.chromadb.get("collection_name", "financial_intel")

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)

        # Use OpenAI embeddings for query
        from langchain_openai import OpenAIEmbeddings
        emb_config = get_settings().embeddings
        embeddings = OpenAIEmbeddings(model=emb_config.model)
        query_embedding = await embeddings.aembed_query(query)

        # Search
        where = filter_metadata if filter_metadata else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i],
            })

        return formatted

    except Exception as e:
        return [{"error": f"Vector search failed: {str(e)}"}]


# ============================================================================
# Vendor API Tool (Mock/Placeholder)
# ============================================================================

class VendorAPIArgs(BaseModel):
    vendor_name: str = Field(description="Vendor name")
    endpoint: str = Field(default="catalog", description="API endpoint: catalog, pricing, partners")


@register_tool(
    category="vendor",
    name="vendor_api",
    description="Query vendor APIs for product catalogs, pricing, partner programs",
    tags=["vendor", "api", "product-data"],
)
async def vendor_api(vendor_name: str, endpoint: str = "catalog") -> Dict[str, Any]:
    """Query vendor API (mock implementation - replace with real APIs)."""
    settings = get_settings()
    base_url = settings.tools.vendor_api.get("base_url")
    api_key = settings.tools.vendor_api.get("api_key")

    if not base_url:
        # Return mock data for development
        return _mock_vendor_data(vendor_name, endpoint)

    async with httpx.AsyncClient(timeout=30) as client:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = await client.get(
            f"{base_url}/vendors/{vendor_name.lower()}/{endpoint}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


def _mock_vendor_data(vendor_name: str, endpoint: str) -> Dict[str, Any]:
    """Mock vendor data for development."""
    mock_catalogs = {
        "dell": {
            "catalog": {
                "products": [
                    {"name": "PowerEdge R760", "category": "Servers", "price_range": "$10K-$50K"},
                    {"name": "Latitude 9440", "category": "Laptops", "price_range": "$2K-$3K"},
                    {"name": "OptiPlex 7010", "category": "Desktops", "price_range": "$800-$1500"},
                ],
                "categories": ["Servers", "Storage", "Networking", "PCs", "Workstations"],
            },
            "pricing": {"model": "tiered", "discounts": "volume-based"},
            "partners": {"program": "Dell Technologies Partner Program", "tiers": ["Authorized", "Gold", "Platinum", "Titanium"]},
        },
        "cisco": {
            "catalog": {
                "products": [
                    {"name": "Catalyst 9000", "category": "Switching", "price_range": "$5K-$100K"},
                    {"name": "Meraki MX", "category": "Security/SD-WAN", "price_range": "$1K-$20K"},
                    {"name": "Webex Suite", "category": "Collaboration", "price_range": "$15-$50/user/mo"},
                ],
                "categories": ["Networking", "Security", "Collaboration", "Data Center", "Cloud"],
            },
            "pricing": {"model": "subscription + hardware", "discounts": "partner-tiered"},
            "partners": {"program": "Cisco Partner Program", "tiers": ["Registered", "Select", "Premier", "Gold"]},
        },
        "microsoft": {
            "catalog": {
                "products": [
                    {"name": "Azure", "category": "Cloud", "price_range": "pay-as-you-go"},
                    {"name": "Microsoft 365", "category": "Productivity", "price_range": "$6-$55/user/mo"},
                    {"name": "Dynamics 365", "category": "Business Apps", "price_range": "$50-$210/user/mo"},
                ],
                "categories": ["Cloud", "AI", "Productivity", "Business Apps", "Security"],
            },
            "pricing": {"model": "consumption + subscription", "discounts": "CSP, EA agreements"},
            "partners": {"program": "Microsoft AI Cloud Partner Program", "tiers": ["Member", "Solutions Partner", "Specializations"]},
        },
    }

    vendor_key = vendor_name.lower()
    if vendor_key in mock_catalogs:
        return mock_catalogs[vendor_key].get(endpoint, {})

    return {"error": f"Mock data not available for {vendor_name}", "endpoint": endpoint}


# ============================================================================
# Tool Exports
# ============================================================================

def get_web_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description="Search the web for current information",
        args_schema=WebSearchArgs,
    )


def get_news_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=news_search,
        name="news",
        description="Search news articles",
        args_schema=NewsSearchArgs,
    )


def get_vector_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=vector_search,
        name="vector_search",
        description="Search vector database for relevant documents",
        args_schema=VectorSearchArgs,
    )


def get_vendor_api_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=vendor_api,
        name="vendor_api",
        description="Query vendor APIs for product catalogs, pricing, partner programs",
        args_schema=VendorAPIArgs,
    )