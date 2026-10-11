"""
MCP Server & Tools for Credit Transfer Assistant.
Exposes tools for searching regulations, evaluating syllabus similarity, and checking rules.
(Giai đoạn 3)
"""

from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
from src.schemas import TransferStatus

# Khởi tạo MCP Server
mcp = FastMCP("CreditTransferAssistantServer")


@mcp.tool()
def search_regulations(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Tìm kiếm thông tin quy chế đào tạo và chuyển đổi tín chỉ học phần.

    Args:
        query: Câu hỏi hoặc từ khóa quy chế cần tra cứu.
        top_k: Số lượng trích dẫn trả về.
    """
    # TODO: Kết nối với RAG pipeline (src.rag.CreditTransferRAG)
    return [
        {
            "document": "Quy chế đào tạo theo hệ thống tín chỉ",
            "section": "Điều 12: Công nhận kết quả học tập và chuyển đổi tín chỉ",
            "content": "Sinh viên được công nhận tín chỉ khi nội dung học phần tương đương từ 80% trở lên và đạt điểm yêu cầu..."
        }
    ]


@mcp.tool()
def calculate_syllabus_similarity(source_syllabus: str, target_syllabus: str) -> Dict[str, Any]:
    """
    So sánh đề cương 2 học phần và tính điểm tương đồng nội dung.

    Args:
        source_syllabus: Đề cương chi tiết học phần đã học (môn nguồn).
        target_syllabus: Đề cương chi tiết học phần xin công nhận (môn đích).
    """
    # TODO: Triển khai so sánh ngữ nghĩa / LLM-based syllabus matcher
    return {
        "similarity_score": 0.85,
        "matched_topics": ["Chương 1: Tổng quan", "Chương 2: Kiến trúc cơ bản"],
        "missing_topics": []
    }


@mcp.tool()
def evaluate_transfer_eligibility(
    source_course_code: str,
    target_course_code: str,
    grade: float,
    similarity_score: float
) -> Dict[str, Any]:
    """
    Đánh giá điều kiện công nhận tín chỉ dựa trên điểm số và độ tương đồng đề cương.

    Args:
        source_course_code: Mã môn học nguồn đã học.
        target_course_code: Mã môn học đích cần chuyển đổi.
        grade: Điểm số môn học nguồn đạt được (thang 10).
        similarity_score: Độ tương đồng nội dung (từ 0.0 đến 1.0).
    """
    is_eligible = grade >= 5.0 and similarity_score >= 0.8
    return {
        "is_eligible": is_eligible,
        "status": TransferStatus.APPROVED.value if is_eligible else TransferStatus.REJECTED.value,
        "note": "Đủ điều kiện công nhận tín chỉ." if is_eligible else "Không thỏa mãn ngưỡng tương đồng (>= 80%) hoặc điểm tối thiểu (>= 5.0)."
    }


if __name__ == "__main__":
    mcp.run()

