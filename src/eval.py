"""
Evaluation Pipeline for Credit Transfer Assistant.
Chạy bộ testset 12 câu (gồm 4 câu no_answer), đo Faithfulness, Rejection Rate,
so sánh Hybrid vs Dense RAG, xuất CSV chi tiết và lưu >= 3 file log chạy thật vào logs/.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent import CreditTransferAgent, save_traces_to_jsonl
from src.schemas import EVAL_CSV_PATH, LOGS_DIR, PROJECT_ROOT, TESTSET_PATH, RunTrace


def load_testset(path: Union[str, Path] = TESTSET_PATH) -> List[Dict[str, Any]]:
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path
    items = []
    for line in target_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def evaluate_run(mode: str, run_tag: str) -> Dict[str, Any]:
    testset = load_testset()
    agent = CreditTransferAgent(retrieval_mode=mode)
    traces: List[RunTrace] = []
    rows_csv: List[Dict[str, Any]] = []

    faithful_count = 0
    no_answer_total = 0
    no_answer_correct = 0
    action_verified_count = 0
    action_total = 0

    for item in testset:
        qid = item["id"]
        query = item["query"]
        expected_no_ans = item["expected_no_answer"]
        expected_kw = item["expected_keywords"]
        req_action = item["requires_action"]

        trace = agent.run(query, run_id=f"{run_tag}_{qid}")
        traces.append(trace)
        ans_lower = trace.final_answer.lower()

        is_rejected = "từ chối" in ans_lower or "no_answer" in ans_lower or "không hỗ trợ" in ans_lower
        kw_hits = sum(1 for kw in expected_kw if kw.lower() in ans_lower)
        is_faithful = (kw_hits >= max(1, len(expected_kw) - 1)) and ("doc000" in ans_lower)

        if is_faithful:
            faithful_count += 1

        if expected_no_ans:
            no_answer_total += 1
            if is_rejected:
                no_answer_correct += 1

        tools_called = [s.action_tool for s in trace.steps]
        has_write_and_verify = (
            "submit_exemption_request" in tools_called and "verify_exemption_status" in tools_called
        )
        if req_action:
            action_total += 1
            if has_write_and_verify:
                action_verified_count += 1

        rows_csv.append({
            "id": qid,
            "mode": mode,
            "category": item["category"],
            "faithful": is_faithful,
            "expected_no_answer": expected_no_ans,
            "rejected_correctly": (is_rejected == expected_no_ans),
            "write_and_verify": has_write_and_verify,
            "steps": len(trace.steps),
            "calls": trace.usage.calls,
            "prompt_tokens": trace.usage.prompt_tokens,
            "completion_tokens": trace.usage.completion_tokens,
            "max_prompt": trace.usage.max_prompt,
        })

    # Dọn dẹp file log cũ của cùng run_tag để thư mục logs/ luôn gọn gàng
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for old_log in LOGS_DIR.glob(f"run_{run_tag}_{mode}_*.jsonl"):
        old_log.unlink(missing_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_{run_tag}_{mode}_{ts}.jsonl"
    save_traces_to_jsonl(traces, log_path)

    return {
        "mode": mode,
        "total_questions": len(testset),
        "faithfulness": round(faithful_count / len(testset) * 100, 2),
        "correct_rejection_rate": round(no_answer_correct / max(1, no_answer_total) * 100, 2),
        "action_verify_rate": round(action_verified_count / max(1, action_total) * 100, 2),
        "total_tokens": sum(t.usage.total_tokens for t in traces),
        "log_file": str(log_path.relative_to(PROJECT_ROOT)),
        "rows": rows_csv
    }


if __name__ == "__main__":
    print("=== RUNNING EVALUATION SUITE (HYBRID VS DENSE) ===")
    res_hybrid_1 = evaluate_run(mode="hybrid", run_tag="eval_r1")
    res_dense = evaluate_run(mode="dense", run_tag="eval_r2")
    res_hybrid_2 = evaluate_run(mode="hybrid", run_tag="eval_r3")

    EVAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(res_hybrid_1["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(res_hybrid_1["rows"])

    print(f"\n[1] HYBRID MODE: Faithfulness = {res_hybrid_1['faithfulness']}% | "
          f"Rejection Rate = {res_hybrid_1['correct_rejection_rate']}% | "
          f"Write->Verify = {res_hybrid_1['action_verify_rate']}% | Log: {res_hybrid_1['log_file']}")
    print(f"[2] DENSE MODE : Faithfulness = {res_dense['faithfulness']}% | "
          f"Rejection Rate = {res_dense['correct_rejection_rate']}% | "
          f"Write->Verify = {res_dense['action_verify_rate']}% | Log: {res_dense['log_file']}")
    print(f"[3] CSV Summary exported to: {EVAL_CSV_PATH.relative_to(PROJECT_ROOT)}")

