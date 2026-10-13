"""
RAG Pipeline for Credit Transfer Assistant.
(Giai đoạn 2: Chunking theo Điều khoản, Dense Search + Hybrid BM25 + Metadata Filter)

Lý do thiết kế Chunking (Yêu cầu 2.5 - 3-5 dòng giải thích):
Văn bản quy chế đào tạo có cấu trúc phân cấp chặt chẽ theo từng "Điều" và "Khoản".
Nếu cắt theo độ dài ký tự cố định (fixed-size chunking), điều kiện nhận chứng chỉ (IELTS/AWS)
và mức tín chỉ quy đổi dễ bị tách rời sang 2 chunk khác nhau, gây đứt gãy ngữ cảnh suy luận.
Do đó, hệ thống áp dụng Semantic/Structure-based Chunking cắt theo tiêu đề cấp 2 (`## `),
đảm bảo mỗi chunk chứa trọn vẹn một quy tắc nghiệp vụ độc lập kèm theo metadata gốc.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from src.schemas import DocumentChunk, DocumentMetadata

load_dotenv()


class CreditTransferRAG:
    """Pipeline tìm kiếm quy chế hỗ trợ Dense, BM25 và Hybrid Search kèm Metadata Filter."""

    def __init__(self, data_dir: str = "data", use_api_embedding: bool = False):
        self.data_dir = Path(data_dir)
        self.use_api_embedding = use_api_embedding and bool(os.getenv("OPENAI_API_KEY"))
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.vocab: Dict[str, int] = {}

        self._load_and_chunk_corpus()
        self._build_indices()

    def _tokenize(self, text: str) -> List[str]:
        """Tách từ cơ bản hỗ trợ cả tiếng Việt và mã học phần (ENG101, IT305, 6.5)."""
        text = text.lower()
        tokens = re.findall(
            r"[a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\.]+",
            text
        )
        return [t.strip(".") for t in tokens if t.strip(".")]

    def _parse_markdown_file(self, file_path: Path) -> tuple[DocumentMetadata, str]:
        """Bóc tách YAML frontmatter metadata và phần thân văn bản."""
        raw_text = file_path.read_text(encoding="utf-8")
        meta_dict = {
            "doc_id": file_path.stem,
            "title": file_path.name,
            "version": "1.0",
            "date": "2024-01-01",
            "owner": "Unknown"
        }

        body = raw_text
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                body = parts[2].strip()
                for line in frontmatter.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta_dict[k.strip()] = v.strip()

        metadata = DocumentMetadata(
            doc_id=meta_dict["doc_id"],
            title=meta_dict["title"],
            version=str(meta_dict["version"]),
            date=str(meta_dict["date"]),
            owner=meta_dict["owner"]
        )
        return metadata, body

    def _load_and_chunk_corpus(self) -> None:
        """Đọc các file .md trong data/ và cắt chunk theo từng mục '## '."""
        self.chunks = []
        if not self.data_dir.exists():
            return

        md_files = sorted(self.data_dir.glob("*.md"))
        for file_path in md_files:
            metadata, body = self._parse_markdown_file(file_path)
            sections = re.split(r"(?=\n## )", "\n" + body)

            chunk_idx = 0
            doc_header = ""
            for sec in sections:
                sec_clean = sec.strip()
                if not sec_clean:
                    continue
                if sec_clean.startswith("# ") and "## " not in sec_clean:
                    doc_header = sec_clean
                    continue

                chunk_idx += 1
                full_content = (
                    f"[{metadata.doc_id} | v{metadata.version} | {metadata.date}]\n"
                    f"{doc_header}\n{sec_clean}".strip()
                )
                chunk = DocumentChunk(
                    chunk_id=f"{metadata.doc_id}_C{chunk_idx:02d}",
                    doc_id=metadata.doc_id,
                    content=full_content,
                    metadata=metadata,
                    score=0.0
                )
                self.chunks.append(chunk)

    def _compute_dense_vectors(self, texts: List[str]) -> np.ndarray:
        """Tính vector biểu diễn văn bản (OpenAI API hoặc Local Character/Word N-gram Embedding)."""
        if self.use_api_embedding:
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
                vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
                return vecs / norms
            except Exception:
                pass

        # Fallback Deterministic Dense Vectorizer (chạy 100% offline)
        if not self.vocab:
            all_tokens = set()
            for t in texts:
                words = self._tokenize(t)
                all_tokens.update(words)
                clean_t = re.sub(r"\s+", " ", t.lower())
                all_tokens.update(clean_t[i:i+3] for i in range(max(0, len(clean_t) - 2)))
            self.vocab = {term: idx for idx, term in enumerate(sorted(all_tokens))}

        vecs = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            words = self._tokenize(t)
            clean_t = re.sub(r"\s+", " ", t.lower())
            trigrams = [clean_t[j:j+3] for j in range(max(0, len(clean_t) - 2))]
            for token in words + trigrams:
                if token in self.vocab:
                    vecs[i, self.vocab[token]] += 1.0

        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    def _build_indices(self) -> None:
        """Khởi tạo chỉ mục BM25 và Dense Embeddings."""
        if not self.chunks:
            return
        tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.dense_embeddings = self._compute_dense_vectors([c.content for c in self.chunks])

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
        alpha: float = 0.5,
        version_filter: Optional[str] = None,
        owner_filter: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Truy xuất top_k đoạn quy chế.
        - mode: 'dense', 'bm25', hoặc 'hybrid' (mặc định: 'hybrid')
        - version_filter: lọc theo phiên bản tài liệu (điểm cộng Metadata Filter)
        """
        if not self.chunks or self.bm25 is None or self.dense_embeddings is None:
            return []

        # 1. Tính điểm Dense Search (Cosine Similarity)
        q_vec = self._compute_dense_vectors([query])[0]
        dense_scores = np.dot(self.dense_embeddings, q_vec)

        # 2. Tính điểm Sparse Search (BM25)
        q_tokens = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)

        # Chuẩn hóa Min-Max về [0, 1] để kết hợp công bằng
        def _normalize(arr: np.ndarray) -> np.ndarray:
            min_v, max_v = float(np.min(arr)), float(np.max(arr))
            if max_v - min_v < 1e-9:
                return np.zeros_like(arr)
            return (arr - min_v) / (max_v - min_v)

        dense_norm = _normalize(dense_scores)
        bm25_norm = _normalize(bm25_scores)

        if mode == "dense":
            final_scores = dense_norm
        elif mode == "bm25":
            final_scores = bm25_norm
        else:
            final_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        # 3. Áp dụng Metadata Filter và gom kết quả
        results: List[DocumentChunk] = []
        for idx, chunk in enumerate(self.chunks):
            if version_filter and chunk.metadata.version != str(version_filter):
                continue
            if owner_filter and chunk.metadata.owner != owner_filter:
                continue

            results.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=round(float(final_scores[idx]), 4)
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    rag = CreditTransferRAG()
    print(f"Loaded {len(rag.chunks)} chunks from corpus.")

    print("\n--- TEST 1: Hybrid Search (IELTS 6.5) ---")
    for c in rag.retrieve("IELTS 6.5 được miễn môn gì?", top_k=2, mode="hybrid"):
        print(f"[{c.chunk_id} | score={c.score} | v{c.metadata.version}] {c.content.splitlines()[2]}")

    print("\n--- TEST 2: Dense Search (AWS Cloud) ---")
    for c in rag.retrieve("Chứng chỉ điện toán đám mây AWS miễn môn nào?", top_k=2, mode="dense"):
        print(f"[{c.chunk_id} | score={c.score} | v{c.metadata.version}] {c.content.splitlines()[2]}")

    print("\n--- TEST 3: Metadata Filter (version=2.0) ---")
    for c in rag.retrieve("IELTS 6.0 miễn môn gì?", top_k=2, mode="hybrid", version_filter="2.0"):
        print(f"[{c.chunk_id} | score={c.score} | v{c.metadata.version}] {c.content.splitlines()[2]}")
