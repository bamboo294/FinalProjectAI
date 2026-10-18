"""
Secondary MCP Client Host (Điểm cộng +1: Chứng minh MCP Interoperability).
"""

import asyncio
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import PROJECT_ROOT


async def main() -> None:
    server_env = os.environ.copy()
    server_env["PYTHONPATH"] = str(PROJECT_ROOT)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        env=server_env,
        cwd=str(PROJECT_ROOT)
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

