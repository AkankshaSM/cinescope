import asyncio
import json
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


async def search_web(query: str) -> dict:
    tools = await get_tavily_tools()

    search_tool = next(
        (
            tool for tool in tools
            if "search" in tool.name.lower()
        ),
        None,
    )

    if search_tool is None:
        raise RuntimeError("Tavily search tool unavailable")

    result = await search_tool.ainvoke({
        "query": query,
    })

    if isinstance(result, list):
        text = "\n".join(
            item.get("text", "")
            for item in result
            if isinstance(item, dict)
            and item.get("type") == "text"
        )
    else:
        text = str(result)

    payload = json.loads(text)

    if "error" in payload:
        raise RuntimeError(
            f"Tavily search failed: {payload.get('detail', payload['error'])}"
        )

    return {
        "query": query,
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in payload.get("results", [])
        ],
    }


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
