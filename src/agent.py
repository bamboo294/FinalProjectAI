"""
Custom Agent Loop for Credit Transfer Assistant.
(Giai đoạn 4: Tự viết vòng lặp có MAX_STEPS, bắt lỗi an toàn, ghi log USAGE)
"""

from datetime import datetime
from pathlib import Path
from src.schemas import AgentStepRecord, RunTrace, TokenUsage

MAX_STEPS = 6


class CreditTransferAgent:
    """Agent điều phối tư vấn và xét miễn môn học / chuyển đổi tín chỉ."""

    def __init__(self, system_prompt_path: str = "prompts/system_prompt.txt"):
        self.system_prompt = self._load_system_prompt(system_prompt_path)

    def _load_system_prompt(self, path_str: str) -> str:
        prompt_path = Path(path_str)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Bạn là trợ lý AI xét miễn môn học và chuyển đổi tín chỉ."

    def run(self, user_query: str) -> RunTrace:
        trace = RunTrace(
            run_id="demo-001",
            timestamp=datetime.now().isoformat(),
            query=user_query,
            final_answer="Khung Agent đã sẵn sàng cho Giai đoạn 2, 3, 4.",
            steps=[
                AgentStepRecord(
                    step_number=1,
                    thought="Kiểm tra kết nối schema thành công.",
                    action_tool="none"
                )
            ],
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0)
        )
        return trace


if __name__ == "__main__":
    agent = CreditTransferAgent()
    result = agent.run("Em có IELTS 6.5 thì được miễn những môn nào theo quy chế 2026?")
    print("Run ID:", result.run_id)
    print("Answer:", result.final_answer)
