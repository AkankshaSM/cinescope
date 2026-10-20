
import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)


def create_mcp_client() -> MultiServerMCPClient:
    """Configure the Tavily Remote MCP connection."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing")

    url = (
        "https://mcp.tavily.com/mcp/"
        f"?tavilyApiKey={quote(api_key, safe='')}"
    )

    return MultiServerMCPClient({
        "tavily": {
            "url": url,
            "transport": "streamable_http",
        }
    })


async def get_tavily_tools():
    """Discover tools exposed by Tavily's MCP server."""

    client = create_mcp_client()
    return await client.get_tools()


async def search_web(query: str) -> str:
    """Execute a real Tavily MCP search."""

    tools = await get_tavily_tools()

    search_tool = next(
        (
            tool for tool in tools
            if "search" in tool.name.lower()
        ),
        None,
    )

    if search_tool is None:
        raise RuntimeError(
            "Tavily MCP server did not expose a search tool"
        )

    result = await search_tool.ainvoke({
        "query": query,
    })

    return str(result)


async def main():
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = "Latest movie release announcements"

    print("Connecting to Tavily Remote MCP...")

    tools = await get_tavily_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print(f"- {tool.name}")

    print(f"\nSearching: {query}\n")

    result = await search_web(query)

    print("=== Tavily Search Results ===\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
