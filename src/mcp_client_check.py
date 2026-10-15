"""
Secondary MCP Client Host (Điểm cộng +1: Chứng minh MCP Interoperability).
"""

import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        env=os.environ.copy()
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("[Host 2] Discovered tools:", [t.name for t in tools.tools])
            res = await session.call_tool("verify_exemption_status", {"ticket_id": "REQ-2026-001"})
            print("[Host 2] Verify response:", [c.text for c in res.content if hasattr(c, "text")])


if __name__ == "__main__":
    asyncio.run(main())

