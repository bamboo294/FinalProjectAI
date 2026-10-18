from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

# Quản lý tập trung toàn bộ đường dẫn của dự án (neo tuyệt đối theo thư mục gốc)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = DATA_DIR / "exemptions_db.json"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
SYSTEM_PROMPT_PATH: Path = PROMPTS_DIR / "system_prompt.txt"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
TESTSET_PATH: Path = TESTS_DIR / "testset.jsonl"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DEMO_LOG_PATH: Path = LOGS_DIR / "run_demo_latest.jsonl"
EVAL_CSV_PATH: Path = LOGS_DIR / "eval_summary.csv"


@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    version: str
    date: str
    owner: str


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: DocumentMetadata
    score: float = 0.0


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
        """Trả về chuỗi kết quả an toàn cho LLM đọc, không raise exception."""
        if not self.success:
            return f"Error: {self.error_message}"
        return f"Success: {self.data}"


@dataclass
class TokenUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    max_prompt: int = 0


@dataclass
class AgentStepRecord:
    step_number: int
    thought: str
    action_tool: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None


@dataclass
class RunTrace:
    run_id: str
    timestamp: str
    query: str
    final_answer: str
    steps: List[AgentStepRecord] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)