"""
Channel Intelligence Agent — Tools Package
"""

from financial_intel.tools.registry import (
    ToolRegistry,
    ToolMetadata,
    get_registry,
    register_tool,
    get_tool,
)

from financial_intel.tools.core import (
    web_search,
    news_search,
    vector_search,
    vendor_api,
    get_web_search_tool,
    get_news_tool,
    get_vector_search_tool,
    get_vendor_api_tool,
)

# Auto-register core tools on import
def register_core_tools() -> ToolRegistry:
    """Register all core tools in the global registry."""
    registry = get_registry()

    # Register if not already registered
    if not registry.get("web_search"):
        registry.register(get_web_search_tool(), ToolMetadata(
            name="web_search",
            description="Search the web for current information using Tavily or SerpAPI",
            category="search",
            tags=["web", "search", "real-time"],
        ))

    if not registry.get("news"):
        registry.register(get_news_tool(), ToolMetadata(
            name="news",
            description="Search news articles using NewsAPI",
            category="search",
            tags=["news", "real-time", "market-intelligence"],
        ))

    if not registry.get("vector_search"):
        registry.register(get_vector_search_tool(), ToolMetadata(
            name="vector_search",
            description="Search vector database for relevant documents",
            category="retrieval",
            tags=["rag", "vector", "semantic-search"],
        ))

    if not registry.get("vendor_api"):
        registry.register(get_vendor_api_tool(), ToolMetadata(
            name="vendor_api",
            description="Query vendor APIs for product catalogs, pricing, partner programs",
            category="vendor",
            tags=["vendor", "api", "product-data"],
        ))

    return registry


# Register on import
register_core_tools()

__all__ = [
    "ToolRegistry",
    "ToolMetadata",
    "get_registry",
    "register_tool",
    "get_tool",
    "web_search",
    "news_search",
    "vector_search",
    "vendor_api",
    "register_core_tools",
]