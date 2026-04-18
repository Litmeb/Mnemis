from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from mnemis_build.cli import _parse_user_index_list, _resolve_user_indexes
from mnemis_build.config import BuildConfig
from mnemis_build.llm import OpenAILLMClient
from mnemis_build.logging_utils import configure_logging, get_logger
from mnemis_build.neo4j_store import Neo4jGraphStore
from mnemis_build.retrieval import MnemisRetriever

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UX dependency
    tqdm = None


JUDGE_SYSTEM_PROMPT = """You are an evaluator for long-term memory question answering.

Decide whether the predicted answer correctly answers the question given the reference answer.

Rules:
- Be semantically lenient: allow paraphrases, minor wording differences, reordered lists, and normalized dates.
- Mark correct when the prediction preserves the essential facts in the reference answer.
- Mark incorrect when it misses a required fact, adds a contradiction, names the wrong entity, or gives the wrong time/place/value.
- If the reference answer contains multiple items, the prediction must include all essential items to be correct.
- Respond with valid json only.
"""


class JudgeResult(BaseModel):
    is_correct: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_result_shape(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "is_correct" in value:
            return value
        result = str(value.get("result", "")).strip().lower()
        if result in {"correct", "incorrect"}:
            return {
                "is_correct": result == "correct",
                "confidence": float(value.get("confidence", 0.5) or 0.5),
                "reasoning": str(value.get("reasoning", result)),
            }
        return value


@dataclass(slots=True)
class EvalQuestion:
    user_index: int
    group_id: str
    question_index: int
    question: str
    reference_answer: str
    evidence: list[str]
    category: int | str | None


class AccuracyTracker:
    def __init__(self) -> None:
        self.completed = 0
        self.correct = 0
        self.failed = 0
        self._lock = asyncio.Lock()

    async def record(self, *, is_correct: bool, failed: bool) -> tuple[int, int, int]:
        async with self._lock:
            self.completed += 1
            if is_correct:
                self.correct += 1
            if failed:
                self.failed += 1
            return self.completed, self.correct, self.failed


def _normalize_reference_answer(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _load_eval_questions(data_path: Path, *, group_id_prefix: str, selected_user_indexes: list[int] | None) -> list[EvalQuestion]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    user_indexes = _resolve_user_indexes(selected_user_indexes, total_users=len(data))
    questions: list[EvalQuestion] = []
    for user_index in user_indexes:
        user = data[user_index]
        group_id = f"{group_id_prefix}_{user_index}"
        for question_index, qa in enumerate(user.get("qa", [])):
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            questions.append(
                EvalQuestion(
                    user_index=user_index,
                    group_id=group_id,
                    question_index=question_index,
                    question=question,
                    reference_answer=_normalize_reference_answer(qa.get("answer")),
                    evidence=[str(item) for item in qa.get("evidence", [])],
                    category=qa.get("category"),
                )
            )
    return questions


async def _judge_answer(
    llm: OpenAILLMClient,
    *,
    question: str,
    reference_answer: str,
    predicted_answer: str,
) -> JudgeResult:
    return await llm.complete_json(
        JudgeResult,
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "reference_answer": reference_answer,
                        "predicted_answer": predicted_answer,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        stage="evaluation",
        operation="judge_answer",
        use_small_model=True,
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "user_index",
        "group_id",
        "question_index",
        "question",
        "reference_answer",
        "predicted_answer",
        "is_correct",
        "judge_confidence",
        "judge_reasoning",
        "category",
        "evidence",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mnemis on LoCoMo with an LLM judge.")
    parser.add_argument("--data", type=Path, default=Path("data/locomo.json"))
    parser.add_argument("--group-id-prefix", default="locomo_user")
    parser.add_argument(
        "--user-index",
        dest="user_indexes",
        type=_parse_user_index_list,
        help="Optional comma-separated list of user indexes to evaluate, for example 0,1,4.",
    )
    parser.add_argument("--max-concurrent-questions", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory. Defaults to the same timestamped run directory as the log file.",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    log_path = configure_logging()
    logger = get_logger("eval")
    config = BuildConfig.from_env()
    output_dir = args.output_dir or log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = _load_eval_questions(
        args.data,
        group_id_prefix=args.group_id_prefix,
        selected_user_indexes=args.user_indexes,
    )
    if not questions:
        raise RuntimeError("No LoCoMo QA items found for the selected users.")

    logger.info(
        "evaluation start | data=%s, question_count=%s, max_concurrent_questions=%s, output_dir=%s",
        args.data,
        len(questions),
        args.max_concurrent_questions,
        output_dir,
    )

    answers_path = output_dir / "locomo_eval_answers.jsonl"
    store = Neo4jGraphStore(config)
    llm = OpenAILLMClient(config)
    retriever = MnemisRetriever(store, llm, config)
    semaphore = asyncio.Semaphore(max(1, args.max_concurrent_questions))
    tracker = AccuracyTracker()
    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(questions), desc="LoCoMo eval", unit="q", dynamic_ncols=True) if tqdm else None

    async def evaluate_one(item: EvalQuestion) -> dict[str, Any]:
        async with semaphore:
            error_text: str | None = None
            predicted_answer = ""
            judge_payload: JudgeResult | None = None
            try:
                answer_payload = await retriever.answer(item.question, item.group_id)
                predicted_answer = answer_payload["answer"]
                judge_payload = await _judge_answer(
                    llm,
                    question=item.question,
                    reference_answer=item.reference_answer,
                    predicted_answer=predicted_answer,
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "evaluation item failed | user_index=%s, group_id=%s, question_index=%s",
                    item.user_index,
                    item.group_id,
                    item.question_index,
                )

            row = {
                "user_index": item.user_index,
                "group_id": item.group_id,
                "question_index": item.question_index,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "predicted_answer": predicted_answer,
                "is_correct": bool(judge_payload and judge_payload.is_correct),
                "judge_confidence": judge_payload.confidence if judge_payload else 0.0,
                "judge_reasoning": judge_payload.reasoning if judge_payload else "",
                "category": item.category,
                "evidence": json.dumps(item.evidence, ensure_ascii=False),
                "error": error_text or "",
            }

            completed, correct, failed = await tracker.record(
                is_correct=row["is_correct"],
                failed=error_text is not None,
            )
            accuracy = correct / completed if completed else 0.0
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(
                    f"acc={correct}/{completed} ({accuracy:.2%}) failed={failed}"
                )
            else:
                print(
                    f"[eval] {completed}/{len(questions)} acc={correct}/{completed} ({accuracy:.2%}) "
                    f"failed={failed} user={item.user_index} q={item.question_index}"
                )
            return row

    try:
        await store.ensure_indexes()
        gathered = await asyncio.gather(*(evaluate_one(item) for item in questions))
    finally:
        if progress is not None:
            progress.close()
        await store.close()

    results.extend(sorted(gathered, key=lambda row: (row["user_index"], row["question_index"])))

    with answers_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "locomo_eval_answers.csv"
    _write_csv(results, csv_path)

    correct = sum(1 for row in results if row["is_correct"])
    failed = sum(1 for row in results if row["error"])
    summary = {
        "question_count": len(results),
        "correct_count": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "failed_count": failed,
        "data_path": str(args.data),
        "group_id_prefix": args.group_id_prefix,
        "max_concurrent_questions": max(1, args.max_concurrent_questions),
        "log_path": str(log_path),
        "answers_jsonl": str(answers_path),
        "answers_csv": str(csv_path),
    }
    summary_path = output_dir / "locomo_eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("evaluation done | summary=%s", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
