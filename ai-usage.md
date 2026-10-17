# Khai Báo Sử Dụng AI (Academic Integrity & AI Usage Declaration)

Dự án tuân thủ quy định liêm chính học thuật của môn **Lập trình AI với Python (AIPR)**. Dưới đây là khai báo ngắn gọn, minh bạch các phần công việc có sử dụng AI hỗ trợ và phần do sinh viên tự thiết kế, kiểm soát:
Model sử dụng: Gemini Pro 3.1 và Gemini Flash 3.8
---

## 1. Tạo dữ liệu & văn bản quy chế (`data/*.md`)
- **Mức độ sử dụng**: Dùng AI tạo hoàn toàn nội dung các file dữ liệu quy chế mẫu dựa trên yêu cầu nghiệp vụ của đề tài.
- **File do AI tạo**:
  - `data/DOC0007_EXEMPT_2024.md` (Quy chế xét miễn năm 2024 - `v1.0`)
  - `data/DOC0008_EXEMPT_2026.md` (Quy chế xét miễn năm 2026 - `v2.0`)
  - `data/DOC0009_CURRICULUM_IT.md` (Khung chương trình CNTT - `v1.0`)
  - `data/DOC0010_EXEMPT_TOEIC_TOEFL.md` (Quy định bổ sung TOEIC/TOEFL theo lộ trình 2026–2027 - `v2.0`)

---

## 2. Khởi tạo môi trường cho Project (Environment Setup)
- **Mức độ sử dụng**: Dùng AI hỗ trợ khởi tạo môi trường ảo Python 3.11 (`venv`), cấu hình `.python-version` và khai báo danh sách thư viện chuẩn trong `requirements.txt`.

---

## 3. Boilerplate & Config (`.env.example`, `Makefile`, `.gitignore`, `src/schemas.py`)
- **Mô tả & Prompt sử dụng**: *"Phần khai báo Data Class, cấu trúc thư mục và file config em đã dùng AI để sinh tự động theo hợp đồng dữ liệu chuẩn."*
- **File & Block code do AI tạo**: Các file cấu hình `.gitignore`, `.env.example`, `Makefile` và các class `@dataclass` trong `src/schemas.py` (dòng 5–44):
```python
# src/schemas.py (Dòng 23-44)
@dataclass
class ExemptionRequestInput:
    student_id: str
    cert_name: str
    cert_score: float
    issue_date: str
    course_codes: List[str]
    notes: Optional[str] = None

@dataclass
class ToolExecutionResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_agent_string(self) -> str:
        if not self.success:
            return f"Error: {self.error_message}"
        return f"Success: {self.data}"
```

---

## 4. Bóc tách YAML & File Utility (`src/rag.py`)
- **Mô tả & Prompt sử dụng**: *"Phần đọc file và regex bóc tách YAML Header sơ cấp em nhờ AI viết helper function."*
- **Block code do AI tạo**: Hàm helper `_parse_markdown_file` trong `src/rag.py` (dòng 56–84):
```python
# src/rag.py (Dòng 66-76)
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
```

---

## 5. MCP Server Setup sơ cấp (`FastMCP` boilerplate trong `src/mcp_server.py`)
- **Mô tả & Prompt sử dụng**: *"Khung khởi tạo MCP Server được AI hỗ trợ sinh theo template chuẩn của SDK."*
- **Block code do AI tạo**: Khung khởi tạo `FastMCP`, hàm đọc/ghi file JSON cơ bản (`_load_db`, `_save_db` tại dòng 16–42) và lệnh chạy `stdio` (dòng 266–268) trong `src/mcp_server.py`:
```python
# src/mcp_server.py (Dòng 16-42 & 266-268)
mcp = FastMCP("CreditTransferAssistantServer")
DB_PATH = Path("data/exemptions_db.json")

def _load_db() -> dict:
    if not DB_PATH.exists():
        return {"records": {}}
    return json.loads(DB_PATH.read_text(encoding="utf-8"))

def _save_db(db_data: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db_data, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 6. Structural Chunking (Regex Lookahead trong `src/rag.py`)
- **Mô tả & Phân định đóng góp**: Em là người trực tiếp thiết kế chiến lược cắt chunk theo cấu trúc tiêu đề cấp 2 (`## `) để không làm đứt gãy bảng quy đổi điểm, đồng thời tự quy định phương trình Regex Lookahead `r"(?=\n## )"`. AI chỉ hỗ trợ viết vòng lặp ghép chuỗi metadata vào đầu mỗi chunk trong hàm `_load_and_chunk_corpus` (`src/rag.py`, dòng 91–118):
```python
# src/rag.py (Dòng 93-109) - Phương trình Regex do SV quy định, AI hỗ trợ vòng lặp đóng gói chunk
sections = re.split(r"(?=\n## )", "\n" + body)
for sec in sections:
    sec_clean = sec.strip()
    if not sec_clean or (sec_clean.startswith("# ") and "## " not in sec_clean):
        continue
    chunk_idx += 1
    full_content = (
        f"[{metadata.doc_id} | v{metadata.version} | {metadata.date} | owner:{metadata.owner}]\n"
        f"{doc_header}\n{sec_clean}".strip()
    )
```

---

## 7. Kiểm thử các lỗi nghiệp vụ (Business Logic & Edge-case Testing)
- **Mức độ sử dụng**: Sau khi hoàn thiện luồng chính (Hybrid RAG, ReAct Agent Loop, Write $\rightarrow$ Verify), em dùng AI để sinh nhanh các câu hỏi kiểm thử biên (điểm dưới ngưỡng tối thiểu, quên nhập ngày cấp, chứng chỉ quá hạn 2 năm, nộp trùng đơn nhiều lần, nhập ngày `DD/MM/YYYY`, truy vấn tiếng Việt không dấu) nhằm rà soát lỗi và kiểm chứng độ chính xác của hệ thống.
