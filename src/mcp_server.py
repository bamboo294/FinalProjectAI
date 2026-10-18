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

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import DB_PATH, ExemptionRequestInput, ToolExecutionResult
from src.rag import CreditTransferRAG

mcp = FastMCP("CreditTransferAssistantServer")
rag_engine = CreditTransferRAG()

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

        # Kiểm tra Lỗi 1: Ngưỡng điểm tối thiểu theo quy chế DOC0008 & DOC0010
        c_lower = cert_name.lower()
        if "ielts" in c_lower and cert_score < 6.0:
            return ToolExecutionResult(
                success=False,
                error_message=f"Điểm IELTS {cert_score} chưa đạt ngưỡng tối thiểu 6.0 theo quy chế DOC0008 (v2.0, Điều 1)."
            ).to_agent_string()
        if "toeic" in c_lower and cert_score < 700:
            return ToolExecutionResult(
                success=False,
                error_message=f"Điểm TOEIC {cert_score} chưa đạt ngưỡng tối thiểu 700 theo quy định DOC0010 (v2.0, Điều 1)."
            ).to_agent_string()
        if "toefl" in c_lower and cert_score < 60:
            return ToolExecutionResult(
                success=False,
                error_message=f"Điểm TOEFL {cert_score} chưa đạt ngưỡng tối thiểu 60 theo quy định DOC0010 (v2.0, Điều 1)."
            ).to_agent_string()

        # Kiểm tra Lỗi 3 & Lỗi 5: Hỗ trợ đa định dạng ngày (YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY) và kiểm tra hạn 2 năm
        issued_dt = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                issued_dt = datetime.strptime(issue_date.strip(), fmt).date()
                break
            except ValueError:
                continue

        if issued_dt is None:
            return ToolExecutionResult(
                success=False,
                error_message=f"Định dạng ngày cấp '{issue_date}' không hợp lệ (hỗ trợ YYYY-MM-DD hoặc DD/MM/YYYY)."
            ).to_agent_string()

        normalized_date = issued_dt.strftime("%Y-%m-%d")
        ref_date = datetime(2026, 2, 1).date()
        today_date = datetime.now().date()
        if issued_dt > max(ref_date, today_date):
            return ToolExecutionResult(
                success=False,
                error_message=f"Ngày cấp {normalized_date} không hợp lệ (nằm ở tương lai)."
            ).to_agent_string()
        remaining_days = 730 - (ref_date - issued_dt).days
        if remaining_days < 30:
            return ToolExecutionResult(
                success=False,
                error_message=f"Chứng chỉ cấp ngày {normalized_date} đã hết hạn hoặc còn dưới 30 ngày hiệu lực theo DOC0008 (v2.0, Điều 4)."
            ).to_agent_string()

        if "IT101" in course_codes:
            return ToolExecutionResult(
                success=False,
                error_message="Môn IT101 (Nhập môn lập trình) không áp dụng miễn giảm qua chứng chỉ nghề theo DOC0009."
            ).to_agent_string()

        req = ExemptionRequestInput(
            student_id=student_id.upper(),
            cert_name=cert_name,
            cert_score=float(cert_score),
            issue_date=normalized_date,
            course_codes=course_codes,
            notes=notes
        )

        requested_credits = sum(COURSE_CREDITS.get(code, 3) for code in req.course_codes)
        capped_credits = min(requested_credits, MAX_CREDIT_CAP)
        is_capped = requested_credits > MAX_CREDIT_CAP

        db = _load_db()

        # Kiểm tra Lỗi 4: Chống nộp trùng hồ sơ (Idempotent Check theo student_id + course_codes)
        for existing_id, existing_rec in db.get("records", {}).items():
            if (
                existing_rec.get("student_id", "").upper() == req.student_id
                and sorted(existing_rec.get("course_codes", [])) == sorted(req.course_codes)
            ):
                existing_rec["is_duplicate"] = True
                existing_rec["updated_at"] = datetime.now().isoformat()
                _save_db(db)
                return ToolExecutionResult(success=True, data=existing_rec).to_agent_string()

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
            "is_duplicate": False,
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
                    "cert_name": record.get("cert_name", ""),
                    "is_duplicate": record.get("is_duplicate", False)
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
