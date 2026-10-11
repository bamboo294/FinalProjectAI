# Kê Khai Sử Dụng AI (AI Usage Declaration)

## 1. Thông Tin Chung
- **Dự án**: AIPR Credit Transfer Assistant (Hệ thống trợ lý AI tư vấn và thẩm định chuyển đổi tín chỉ học phần)
- **Học phần**: Trí tuệ nhân tạo & Lập trình Python ứng dụng
- **Tác giả / Nhóm thực hiện**: [Điền tên sinh viên / nhóm]

## 2. Công Cụ AI Đã Sử Dụng
- **Trợ lý lập trình**: Antigravity Coding Assistant (Gemini 3.8 Flash).
- **Mô hình ngôn ngữ lớn (LLM)**: Gemini API (`gemini-2.5-flash`), OpenAI API (`gpt-4o-mini`).
- **Framework & Công nghệ**: Model Context Protocol (MCP), ChromaDB, Sentence-Transformers.

## 3. Danh Mục Các Phần Có Sự Hỗ Trợ Của AI
- **Khởi tạo kiến trúc dự án**: Hỗ trợ tổ chức cấu trúc thư mục module hóa 4 giai đoạn (`src/`, `prompts/`, `tests/`, `logs/`).
- **Thiết kế Data Contracts (`src/schemas.py`)**: Hỗ trợ định nghĩa các `@dataclass` mô hình hóa dữ liệu học phần (`Course`), trích đoạn quy chế (`RegulationChunk`), kết quả thẩm định (`TransferEvaluation`) và bản ghi trace bước chạy (`AgentTraceStep`).
- **Xây dựng Template Mã Nguồn**:
  - `src/rag.py`: Thiết lập cấu trúc lớp `CreditTransferRAG` cho giai đoạn 2.
  - `src/mcp_server.py`: Thiết lập khung MCP Server và các tool chuyên biệt (`search_regulations`, `calculate_syllabus_similarity`, `evaluate_transfer_eligibility`) cho giai đoạn 3.
  - `src/agent.py`: Thiết lập khung điều phối vòng lặp tác tử `CreditTransferAgent` cho giai đoạn 4.
- **Biên soạn System Prompt (`prompts/system_prompt.txt`)**: Định hình nhân vật, quy chuẩn pháp lý và tiêu chí phân loại kết quả thẩm định (APPROVED, REJECTED, CONDITIONAL, UNDER_REVIEW).
- **Thiết kế Bộ Test Mẫu (`tests/testset.jsonl`)**: Sinh các ca kiểm thử mô phỏng các tình huống thực tế sinh viên xin chuyển đổi tín chỉ.

## 4. Cam Kết & Trách Nhiệm Của Tác Giả
- Tác giả đã kiểm tra, rà soát và hiểu rõ toàn bộ cấu trúc mã nguồn được AI hỗ trợ sinh ra.
- Toàn bộ logic nghiệp vụ, các bước kiểm thử và tinh chỉnh tiếp theo do tác giả trực tiếp thực hiện và chịu trách nhiệm.

