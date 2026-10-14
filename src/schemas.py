from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


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