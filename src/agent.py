"""
Custom Agent Loop for Credit Transfer Assistant.
Orchestrates reasoning, tool execution, and final transfer evaluation.
(Giai đoạn 4)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from src.schemas import AgentTraceStep, TransferEvaluation, TransferStatus


class CreditTransferAgent:
    """Agent điều phối quy trình tư vấn và đánh giá chuyển đổi tín chỉ."""

    def __init__(
        self,
        system_prompt_path: str = "prompts/system_prompt.txt",
        model_name: str = "gemini-2.5-flash"
    ):
        self.model_name = model_name
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.history: List[Dict[str, Any]] = []

    def _load_system_prompt(self, path_str: str) -> str:
        prompt_path = Path(path_str)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Bạn là trợ lý AI chuyên trách tư vấn và thẩm định chuyển đổi tín chỉ học phần."

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Thực thi vòng lặp tác tử (Agent Loop):
        1. Nhận yêu cầu từ người dùng.
        2. Suy luận (Thought) & xác định công cụ cần gọi.
        3. Thực thi công cụ (Action) qua MCP tools hoặc RAG.
        4. Quan sát kết quả (Observation).
        5. Lặp lại cho đến khi có kết luận cuối cùng (Final Answer).
        """
        trace: List[AgentTraceStep] = []

        # TODO: Triển khai Agent Loop tích hợp LLM & MCP Tools
        # Ví dụ một bước trace mẫu:
        trace.append(
            AgentTraceStep(
                step_number=1,
                thought="Cần tra cứu quy chế về ngưỡng tương đồng đề cương và điều kiện điểm số.",
                tool_name="search_regulations",
                tool_input={"query": "điều kiện chuyển đổi tín chỉ học phần"},
                tool_output="Yêu cầu điểm >= 5.0 và độ tương đồng >= 80%.",
                observation="Đã tìm thấy căn cứ quy chế liên quan."
            )
        )

        return {
            "query": user_query,
            "response": "Hệ thống đang trong quá trình hoàn thiện Agent Loop (Giai đoạn 4).",
            "trace": [t.__dict__ for t in trace]
        }


if __name__ == "__main__":
    agent = CreditTransferAgent()
    result = agent.run("Tôi muốn chuyển đổi học phần Giải tích 1 từ trường ĐH Bách Khoa sang.")
    print("Agent result:", result)

