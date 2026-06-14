"""
Channel Intelligence Agent — Tool Registry (MCP-Style)

Centralized tool registration with schemas, similar to Model Context Protocol.
Tools are registered with metadata for discovery and invocation.
"""

from typing import Any, Callable, Dict, List, Optional, Type
from functools import wraps
import inspect

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""

    name: str
    description: str
    category: str  # search, retrieval, vendor, partner, market, utility
    version: str = "1.0.0"
    author: str = "channel-intel"
    tags: List[str] = Field(default_factory=list)
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # calls per minute
    timeout: int = 30
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class ToolRegistry:
    """
    MCP-style tool registry for agent tool management.

    Features:
    - Tool registration with metadata
    - Schema validation
    - Category-based discovery
    - Rate limiting
    - Usage tracking
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        self._usage_stats: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        tool: BaseTool,
        metadata: ToolMetadata,
        override: bool = False,
    ) -> None:
        """Register a tool with metadata."""
        if tool.name in self._tools and not override:
            raise ValueError(f"Tool '{tool.name}' already registered. Use override=True to replace.")

        self._tools[tool.name] = tool
        self._metadata[tool.name] = metadata

        # Update category index
        if metadata.category not in self._categories:
            self._categories[metadata.category] = []
        if tool.name not in self._categories[metadata.category]:
            self._categories[metadata.category].append(tool.name)

        # Initialize usage stats
        self._usage_stats[tool.name] = {
            "calls": 0,
            "errors": 0,
            "total_latency_ms": 0,
            "last_called": None,
        }

    def register_function(
        self,
        func: Callable,
        name: str,
        description: str,
        category: str,
        args_schema: Type[BaseModel],
        **metadata_kwargs,
    ) -> StructuredTool:
        """Register a function as a StructuredTool."""
        tool = StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=args_schema,
        )

        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            **metadata_kwargs,
        )

        self.register(tool, metadata)
        return tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata."""
        return self._metadata.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List registered tool names, optionally filtered by category."""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._categories.keys())

    def get_tools_for_agent(self, categories: List[str]) -> List[BaseTool]:
        """Get tools for specific agent categories."""
        tools = []
        for cat in categories:
            for name in self._categories.get(cat, []):
                tools.append(self._tools[name])
        return tools

    def get_schemas(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for LLM function calling."""
        schemas = []
        names = self.list_tools(category)
        for name in names:
            tool = self._tools[name]
            meta = self._metadata[name]
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {},
                },
            }
            schemas.append(schema)
        return schemas

    async def invoke(self, name: str, **kwargs) -> Any:
        """Invoke a tool with usage tracking."""
        import time

        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        meta = self._metadata[name]

        # Check rate limit
        if meta.rate_limit:
            # Simple in-memory rate limiting (per minute)
            pass  # Implement if needed

        start = time.perf_counter()
        try:
            if hasattr(tool, 'ainvoke'):
                result = await tool.ainvoke(kwargs)
            else:
                result = tool.invoke(kwargs)

            # Update stats
            latency = (time.perf_counter() - start) * 1000
            stats = self._usage_stats[name]
            stats["calls"] += 1
            stats["total_latency_ms"] += latency
            stats["last_called"] = time.time()

            return result
        except Exception as e:
            self._usage_stats[name]["errors"] += 1
            raise

    def get_usage_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics."""
        if name:
            return self._usage_stats.get(name, {})
        return self._usage_stats


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get global tool registry (singleton)."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(
    category: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **metadata_kwargs,
):
    """
    Decorator to register a function as a tool.

    Usage:
        @register_tool(category="search", name="web_search")
        async def web_search(query: str) -> List[Dict]: ...
    """
    def decorator(func: Callable) -> Callable:
        registry = get_registry()
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        # Infer schema from function signature
        sig = inspect.signature(func)
        annotations = func.__annotations__

        # Create args schema dynamically
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = annotations.get(param_name, Any)
            default = param.default if param.default != inspect.Parameter.empty else ...
            fields[param_name] = (param_type, default)

        if fields:
            ArgsSchema = type(f"{tool_name}Args", (BaseModel,), {"__annotations__": fields})
        else:
            ArgsSchema = type(f"{tool_name}Args", (BaseModel,), {})

        registry.register_function(
            func=func,
            name=tool_name,
            description=tool_desc,
            category=category,
            args_schema=ArgsSchema,
            **metadata_kwargs,
        )
        return func

    return decorator


# Convenience function for getting tools
def get_tool(name: str) -> Optional[BaseTool]:
    """Get tool by name from global registry."""
    return get_registry().get(name)