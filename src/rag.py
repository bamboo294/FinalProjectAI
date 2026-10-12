"""
RAG Pipeline for Credit Transfer Assistant.
(Giai đoạn 2: Dense + Hybrid BM25 Search)
"""

from pathlib import Path
from typing import List, Optional
from src.schemas import DocumentChunk, DocumentMetadata


class CreditTransferRAG:
    """Pipeline tìm kiếm quy chế xét miễn môn và chuyển đổi tín chỉ."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.chunks: List[DocumentChunk] = []

    def retrieve(
        self, query: str, top_k: int = 3, mode: str = "hybrid", version_filter: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Truy xuất top_k đoạn quy chế (hỗ trợ dense và hybrid BM25)."""
        # Sẽ triển khai chi tiết ở Giai đoạn 2
        return []


if __name__ == "__main__":
    rag = CreditTransferRAG()
    print("CreditTransferRAG initialized successfully.")
