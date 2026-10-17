# Khai Báo Sử Dụng AI (Academic Integrity & AI Usage Declaration)

Dự án tuân thủ quy định liêm chính học thuật của môn Lập trình AI với Python (AIPR). Dưới đây là chi tiết các phần có sự hỗ trợ của công cụ AI:

## 1. Công cụ sử dụng
- **Gemini / Antigravity Coding Assistant**: Hỗ trợ thiết kế khung dataclass, chuẩn hóa docstring và kiểm tra cú pháp MCP SDK.

## 2. Chi tiết các phần sử dụng AI
| Thành phần / File | Mức độ sử dụng | Prompt chính đã dùng | Kiểm soát & Chỉnh sửa của Sinh viên |
| :--- | :--- | :--- | :--- |
| `src/schemas.py` | Gợi ý cấu trúc `@dataclass` | *"Thiết kế các dataclass có type hint cho DocumentChunk, ExemptionRequestInput, ToolExecutionResult và RunTrace"* | Sinh viên tự định nghĩa các trường dữ liệu khớp với bài toán xét miễn tín chỉ và thêm `max_prompt`. |
| `src/rag.py` | Hỗ trợ hàm regex tách câu và chuẩn hóa Min-Max | *"Viết hàm cắt chunk theo tiêu đề cấp 2 ## và chuẩn hóa điểm BM25 + Cosine Similarity"* | Sinh viên tự thiết kế cấu trúc YAML metadata và kiểm thử trọng số `alpha = 0.5`. |
| `src/mcp_server.py` | Tham khảo cú pháp `@mcp.tool()` của FastMCP | *"Cách viết MCP Server qua stdio không in ra stdout và lưu JSON an toàn"* | Sinh viên tự cài đặt quy tắc mức trần 15 tín chỉ (`MAX_CREDIT_CAP = 15`) và cặp tool Write -> Verify. |
| `src/agent.py` & `src/eval.py` | Hỗ trợ cấu trúc `stdio_client` bất đồng bộ | *"Kết nối MCP Client qua stdio_client, discover động list_tools và ghi log jsonl"* | Sinh viên tự kiểm soát vòng lặp `MAX_STEPS = 6`, hàm `_safe_parse_action` và bộ 12 câu hỏi trong `tests/testset.jsonl`. |
