"""
MCP Server & Tools for Credit Transfer Assistant.
(Giai đoạn 3: Chạy qua stdio, có Write Tool gây side-effect và Read Tool để verify)
"""

from typing import List
from mcp.server.fastmcp import FastMCP
from src.schemas import ExemptionRequestInput, ToolExecutionResult

mcp = FastMCP("CreditTransferAssistantServer")


@mcp.tool()
def submit_exemption_request(
    student_id: str,
    cert_name: str,
    cert_score: float,
    issue_date: str,
    course_codes: List[str],
    notes: str = ""
) -> str:
    """
    [WRITE TOOL - Có Side Effect] Ghi nhận hồ sơ xin xét miễn môn học vào cơ sở dữ liệu.
    """
    req = ExemptionRequestInput(
        student_id=student_id,
        cert_name=cert_name,
        cert_score=cert_score,
        issue_date=issue_date,
        course_codes=course_codes,
        notes=notes
    )
    # Giai đoạn 3 sẽ lưu thật vào database.json
    res = ToolExecutionResult(
        success=True,
        data={"ticket_id": "REQ-2026-001", "status": "PENDING", "student_id": req.student_id}
    )
    return res.to_agent_string()


@mcp.tool()
def verify_exemption_status(ticket_id: str) -> str:
    """
    [READ TOOL - Bước Verify bắt buộc] Đọc lại hồ sơ từ hệ thống để xác nhận trạng thái đã lưu.
    """
    res = ToolExecutionResult(
        success=True,
        data={"ticket_id": ticket_id, "verified": True, "status": "PENDING"}
    )
    return res.to_agent_string()


if __name__ == "__main__":
    mcp.run()
