# AIPR Credit Transfer Assistant

Hệ thống trợ lý AI hỗ trợ tư vấn, thẩm định và xét duyệt chuyển đổi / công nhận tín chỉ học phần dựa trên quy chế đào tạo đại học, áp dụng kiến trúc **RAG**, chuẩn **Model Context Protocol (MCP)** và **Custom Agent Loop**.

---

## 🏛️ Kiến Trúc Hệ Thống & Các Giai Đoạn

Dự án được xây dựng tuần tự theo 4 giai đoạn chuẩn:

```
[ Sinh viên gửi yêu cầu / Hồ sơ xét miễn ]
                 │
                 ▼
     ┌───────────────────────┐
     │  Custom Agent Loop    │ ◄── Giai đoạn 4 (`src/agent.py`)
     │  (Suy luận & Điều phối)│
     └───────────┬───────────┘
                 │
        gọi tools qua MCP (stdio)
                 ▼
     ┌───────────────────────┐
     │      MCP Server       │ ◄── Giai đoạn 3 (`src/mcp_server.py`)
     │  (Các tools chuẩn hóa)│
     └───────────┬───────────┘
                 │
                 ├──────────────────────────────┐
                 ▼                              ▼
     ┌───────────────────────┐      ┌───────────────────────┐
     │     RAG Pipeline      │      │     Exemption DB      │
     │ (`src/rag.py`)        │      │   (Write & Verify)    │
     │ ◄── Giai đoạn 2       │      │(`data/exemptions_db`) │
     └───────────┬───────────┘      └───────────────────────┘
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
   - Phân đoạn theo điều khoản (`## `), hỗ trợ **Dense Search**, **BM25**, **Hybrid Search** và **Metadata Filter** (`version_filter`).
3. **Giai đoạn 3 - MCP Server & Tools (`src/mcp_server.py`)**:
   - Chạy qua `stdio` (stdout sạch tuyệt đối), cung cấp 3 tools: `search_regulations`, `submit_exemption_request` (Write Tool có side-effect lưu vào `data/exemptions_db.json` kèm kiểm tra trần 15 tín chỉ) và `verify_exemption_status` (Read Tool để verify).
4. **Giai đoạn 4 - Custom Agent Loop & Evaluation (`src/agent.py`, `src/eval.py`)**:
   - Vòng lặp tác tử tự viết có `MAX_STEPS = 6`, dynamic tool discovery (`session.list_tools()`), parse JSON an toàn, ép cặp Write -> Verify và ghi log `USAGE` chi tiết vào `logs/`.

---

## 📁 Cấu Trúc Dự Án

```
aipr-credit-transfer-assistant/
├── data/                     # Chứa 3 file tài liệu quy chế (corpus) & exemptions_db.json
├── src/                      # Toàn bộ mã nguồn Python
│   ├── __init__.py
│   ├── schemas.py            # Khai báo @dataclass cho data contracts & TokenUsage
│   ├── rag.py                # Pipeline tìm kiếm (Dense + BM25 + Hybrid + Metadata Filter)
│   ├── mcp_server.py         # MCP Server qua stdio (Search + Write + Verify tools)
│   ├── agent.py              # Custom Agent Loop (Dynamic tool discovery, MAX_STEPS=6)
│   ├── eval.py               # Script đánh giá 12 câu testset, so sánh Hybrid vs Dense & xuất CSV
│   └── mcp_client_check.py   # Client host thứ 2 kiểm chứng khả năng tương tác MCP
├── prompts/                  # Prompt hệ thống tách riêng khỏi code
│   └── system_prompt.txt
├── tests/                    # Bộ testset kiểm thử (12 câu, gồm 4 câu no_answer)
│   └── testset.jsonl
├── logs/                     # Chứa >= 3 file trace .jsonl chạy thật & bảng eval_summary.csv
├── Makefile                  # Lệnh chạy tự động (setup, demo, eval, interop, server)
├── requirements.txt          # Danh sách thư viện
├── .env.example              # File mẫu chứa API key
├── ai-usage.md               # Kê khai sử dụng AI minh bạch
└── README.md                 # Hướng dẫn chạy và kiến trúc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Nhanh (Cold Start 1 Lệnh)

### Chạy nhanh bằng 1 lệnh duy nhất (Yêu cầu 2.1):
```bash
make setup && make demo
```

### 1. Cài đặt môi trường
Khởi tạo môi trường ảo Python và cài đặt các phụ thuộc:

```bash
# Tạo và kích hoạt virtual environment
python -m venv venv

# Kích hoạt trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện:
make setup
# Hoặc: pip install -r requirements.txt
```

### 2. Thiết lập biến môi trường
Sao chép `.env.example` thành `.env` (hệ thống hỗ trợ cả chạy với OpenAI API hoặc chạy 100% offline deterministic khi chấm Cold Start):
```bash
cp .env.example .env
```

### 3. Các lệnh vận hành & Tái lập kết quả (`Makefile`)
- **Chạy thử Agent demo**:
  ```bash
  make demo
  # Hoặc: python -m src.agent
  ```
- **Chạy bộ kiểm thử 12 câu & Tái lập toàn bộ log + CSV (`logs/eval_summary.csv`)**:
  ```bash
  make eval
  # Hoặc: python -m src.eval
  ```
- **Kiểm tra MCP Server từ Client Host thứ hai**:
  ```bash
  make interop
  # Hoặc: python -m src.mcp_client_check
  ```
- **Khởi động MCP Server độc lập qua stdio**:
  ```bash
  make server
  # Hoặc: python -m src.mcp_server
  ```
