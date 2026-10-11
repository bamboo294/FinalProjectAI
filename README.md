# AIPR Credit Transfer Assistant

Hệ thống trợ lý AI hỗ trợ tư vấn, thẩm định và xét duyệt chuyển đổi / công nhận tín chỉ học phần dựa trên quy chế đào tạo đại học, áp dụng kiến trúc **RAG**, chuẩn **Model Context Protocol (MCP)** và **Custom Agent Loop**.

---

## 🏛️ Kiến Trúc Hệ Thống & Các Giai Đoạn

Dự án được xây dựng tuần tự theo 4 giai đoạn chuẩn:

```
[ Sinh viên gửi yêu cầu / Đề cương ]
                 │
                 ▼
     ┌───────────────────────┐
     │  Custom Agent Loop    │ ◄── Giai đoạn 4 (`src/agent.py`)
     │  (Suy luận & Điều phối)│
     └───────────┬───────────┘
                 │
        gọi tools qua MCP
                 ▼
     ┌───────────────────────┐
     │      MCP Server       │ ◄── Giai đoạn 3 (`src/mcp_server.py`)
     │  (Các tools chuẩn hóa)│
     └───────────┬───────────┘
                 │
                 ├──────────────────────────────┐
                 ▼                              ▼
     ┌───────────────────────┐      ┌───────────────────────┐
     │     RAG Pipeline      │      │ Syllabus Matcher      │
     │ (`src/rag.py`)        │      │ (Đối sánh đề cương)   │
     │ ◄── Giai đoạn 2       │      └───────────────────────┘
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ Corpus Quy chế (data/)│ ◄── Giai đoạn 1 (`data/`, `src/schemas.py`)
     └───────────────────────┘
```

1. **Giai đoạn 1 - Chuẩn bị & Data Contracts**:
   - Khởi tạo thư mục và các hợp đồng dữ liệu (`src/schemas.py`) bằng `@dataclass`.
   - Thu thập 3 file quy chế đào tạo vào thư mục `data/`.
   - Thiết lập `prompts/system_prompt.txt` và bộ ca kiểm thử `tests/testset.jsonl`.
2. **Giai đoạn 2 - RAG Pipeline (`src/rag.py`)**:
   - Phân đoạn (chunking), đánh chỉ mục (vector indexing) văn bản quy chế.
   - Truy xuất thông tin (retrieval) chính xác theo điều khoản.
3. **Giai đoạn 3 - MCP Server & Tools (`src/mcp_server.py`)**:
   - Đóng gói các chức năng thành tool của Model Context Protocol.
   - Bao gồm: tìm kiếm quy chế, tính độ tương đồng đề cương, đánh giá điều kiện tiên quyết.
4. **Giai đoạn 4 - Custom Agent Loop (`src/agent.py`)**:
   - Vòng lặp tác tử tự trị (Thought -> Action -> Observation).
   - Tự động gọi tool, phân tích kết quả và xuất báo cáo thẩm định, lưu trace vào `logs/`.

---

## 📁 Cấu Trúc Dự Án

```
aipr-credit-transfer-assistant/
├── data/                  # Chứa 3 file tài liệu quy chế (corpus)
├── src/                   # Toàn bộ mã nguồn Python
│   ├── __init__.py
│   ├── schemas.py         # Khai báo @dataclass cho data contracts
│   ├── rag.py             # Pipeline tìm kiếm (xây ở giai đoạn 2)
│   ├── mcp_server.py      # MCP Server & tools (xây ở giai đoạn 3)
│   └── agent.py           # Custom Agent Loop (xây ở giai đoạn 4)
├── prompts/               # Prompt hệ thống tách riêng khỏi code
│   └── system_prompt.txt
├── tests/                 # Bộ testset kiểm thử
│   └── testset.jsonl
├── logs/                  # Chứa các file trace jsonl khi chạy thật
├── Makefile               # Lệnh chạy tự động (setup, demo, eval)
├── requirements.txt       # Danh sách thư viện
├── .env.example           # File mẫu chứa API key
├── ai-usage.md            # Kê khai sử dụng AI
└── README.md              # Hướng dẫn chạy và kiến trúc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt môi trường
Khởi tạo môi trường ảo Python và cài đặt các phụ thuộc:

```bash
# Tạo và kích hoạt virtual environment
python -m venv venv

# Kích hoạt trên Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Hoặc trên Command Prompt:
# .\venv\Scripts\activate.bat

# Cài đặt thư viện:
make setup
# Hoặc: pip install -r requirements.txt
```

### 2. Thiết lập biến môi trường
Sao chép `.env.example` thành `.env` và cập nhật khóa API:
```bash
cp .env.example .env
```
Cấu hình các API key:
```env
GEMINI_API_KEY="your-gemini-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

### 3. Vận hành hệ thống
Sử dụng `Makefile` để thực thi nhanh các tác vụ:
- **Chạy thử Agent demo**:
  ```bash
  make demo
  ```
- **Khởi động MCP Server**:
  ```bash
  make server
  ```
- **Chạy kiểm thử đánh giá**:
  ```bash
  make eval
  ```

