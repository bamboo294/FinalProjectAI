"""
RAG Pipeline for Credit Transfer Assistant.
Handles indexing, vector embedding, and retrieval of university regulations.
(Giai đoạn 2)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from src.schemas import RegulationChunk


class CreditTransferRAG:
    """Pipeline tìm kiếm và truy xuất ngữ cảnh quy chế đào tạo & chuyển đổi tín chỉ."""

    def __init__(self, data_dir: str = "data", index_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.index_path = Path(index_path) if index_path else None
        self._is_indexed = False

    def load_documents(self) -> List[Dict[str, Any]]:
        """Tải các file quy chế (PDF, Markdown, TXT) trong thư mục data/."""
        # TODO: Triển khai đọc tài liệu quy chế từ data/
        pass

    def build_index(self) -> None:
        """Tạo vector embeddings và xây dựng search index."""
        # TODO: Triển khai embedding và vector store
        pass

    def retrieve(self, query: str, top_k: int = 3) -> List[RegulationChunk]:
        """
        Truy xuất top_k đoạn quy chế liên quan nhất đến câu truy vấn.

        Args:
            query: Câu hỏi hoặc nội dung quy chế cần tra cứu.
            top_k: Số lượng trích dẫn trả về.
        """
        # TODO: Triển khai similarity search / hybrid search
        return []


if __name__ == "__main__":
    rag = CreditTransferRAG()
    print("CreditTransferRAG template initialized successfully.")

