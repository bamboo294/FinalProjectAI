"""
MCP Server & Tools for Credit Transfer Assistant.
(Giai đoạn 3: Chạy qua stdio, stdout sạch tuyệt đối, có Write Tool gây side-effect và Read Tool để verify)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from mcp.server.fastmcp import FastMCP

from src.schemas import ExemptionRequestInput, ToolExecutionResult
from src.rag import CreditTransferRAG

mcp = FastMCP("CreditTransferAssistantServer")
rag_engine = CreditTransferRAG()
DB_PATH = Path("data/exemptions_db.json")

COURSE_CREDITS = {
    "ENG101": 3,
    "ENG102": 3,
    "ENG103": 3,
    "IT305": 3,
    "IT405": 3,
}
MAX_CREDIT_CAP = 15


def _load_db() -> dict:
    if not DB_PATH.exists():
        return {"records": {}}
    try:
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"records": {}}


def _save_db(db_data: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db_data, ensure_ascii=False, indent=2), encoding="utf-8")


@mcp.tool()
def search_regulations(
    query: str,
    top_k: int = 3,
    mode: str = "hybrid",
    version_filter: Optional[str] = None
) -> str:
    """
    Tra cứu quy chế xét miễn môn học và khung chương trình đào tạo.
    Hỗ trợ mode: 'hybrid', 'dense', 'bm25' và lọc theo version_filter ('1.0' hoặc '2.0').
    """
    try:
        chunks = rag_engine.retrieve(
            query=query, top_k=top_k, mode=mode, version_filter=version_filter
        )
        formatted = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "version": c.metadata.version,
                "date": c.metadata.date,
                "score": c.score,
                "content": c.content,
            }
            for c in chunks
        ]
        return ToolExecutionResult(success=True, data={"chunks": formatted}).to_agent_string()
    except Exception as e:
        sys.stderr.write(f"[MCP Error] search_regulations: {e}\n")
        return ToolExecutionResult(success=False, error_message=str(e)).to_agent_string()


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
    [WRITE TOOL - Side Effect] Ghi hồ sơ xét miễn môn học vào data/exemptions_db.json.
    Kiểm tra ngày cấp, mã môn hợp lệ và áp dụng mức trần tối đa 15 tín chỉ (DOC0008 Điều 3).
    """
    try:
        if not student_id or not cert_name or not issue_date or cert_score <= 0:
            return ToolExecutionResult(
                success=False,
                error_message="Thiếu dữ liệu bắt buộc (student_id, cert_name, cert_score > 0, issue_date)."
            ).to_agent_string()

        if "IT101" in course_codes:
            return ToolExecutionResult(
                success=False,
                error_message="Môn IT101 (Nhập môn lập trình) không áp dụng miễn giảm qua chứng chỉ nghề theo DOC0009."
            ).to_agent_string()

        req = ExemptionRequestInput(
            student_id=student_id,
            cert_name=cert_name,
            cert_score=float(cert_score),
            issue_date=issue_date,
            course_codes=course_codes,
            notes=notes
        )

        requested_credits = sum(COURSE_CREDITS.get(code, 3) for code in req.course_codes)
        capped_credits = min(requested_credits, MAX_CREDIT_CAP)
        is_capped = requested_credits > MAX_CREDIT_CAP

        db = _load_db()
        ticket_id = f"REQ-2026-{len(db['records']) + 1:03d}"
        record = {
            "ticket_id": ticket_id,
            "student_id": req.student_id,
            "cert_name": req.cert_name,
            "cert_score": req.cert_score,
            "issue_date": req.issue_date,
            "course_codes": req.course_codes,
            "requested_credits": requested_credits,
            "approved_credits": capped_credits,
            "capping_applied": is_capped,
            "status": "PENDING_REVIEW",
            "created_at": datetime.now().isoformat(),
            "notes": req.notes or ""
        }
        db["records"][ticket_id] = record
        _save_db(db)

        return ToolExecutionResult(success=True, data=record).to_agent_string()
    except Exception as e:
        sys.stderr.write(f"[MCP Error] submit_exemption_request: {e}\n")
        return ToolExecutionResult(success=False, error_message=str(e)).to_agent_string()


@mcp.tool()
def verify_exemption_status(ticket_id: str = "", student_id: str = "") -> str:
    """
    [READ TOOL - Bước Verify bắt buộc] Đọc lại từ data/exemptions_db.json để xác minh hồ sơ đã được ghi thật.
    Hỗ trợ tra cứu theo ticket_id (mã hồ sơ) hoặc student_id (mã sinh viên).
    """
    try:
        db = _load_db()
        records = db.get("records", {})

        if ticket_id:
            if ticket_id not in records:
                return ToolExecutionResult(
                    success=False,
                    error_message=f"Không tìm thấy hồ sơ {ticket_id} trong cơ sở dữ liệu."
                ).to_agent_string()

            record = records[ticket_id]
            return ToolExecutionResult(
                success=True,
                data={
                    "verified": True,
                    "ticket_id": ticket_id,
                    "status": record["status"],
                    "student_id": record["student_id"],
                    "approved_credits": record["approved_credits"],
                    "course_codes": record["course_codes"],
                    "cert_name": record.get("cert_name", "")
                }
            ).to_agent_string()

        if student_id:
            s_up = student_id.upper()
            matched = [r for r in records.values() if r.get("student_id", "").upper() == s_up]
            if not matched:
                return ToolExecutionResult(
                    success=False,
                    error_message=f"Chưa có hồ sơ xét miễn nào cho sinh viên {student_id} trong cơ sở dữ liệu."
                ).to_agent_string()

            latest = matched[-1]
            return ToolExecutionResult(
                success=True,
                data={
                    "verified": True,
                    "ticket_id": latest["ticket_id"],
                    "status": latest["status"],
                    "student_id": latest["student_id"],
                    "approved_credits": latest["approved_credits"],
                    "course_codes": latest["course_codes"],
                    "cert_name": latest.get("cert_name", ""),
                    "total_tickets": len(matched)
                }
            ).to_agent_string()

        return ToolExecutionResult(
            success=False,
            error_message="Cần cung cấp ít nhất ticket_id hoặc student_id để tra cứu hồ sơ."
        ).to_agent_string()
    except Exception as e:
        sys.stderr.write(f"[MCP Error] verify_exemption_status: {e}\n")
        return ToolExecutionResult(success=False, error_message=str(e)).to_agent_string()


if __name__ == "__main__":
    mcp.run()
