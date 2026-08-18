from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_BASE_URL, LLM_EMBEDDING_MODEL, LLM_MODEL, OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # Implementation outline retained for lab review:
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    empty = {name: 0.0 for name in metric_names}
    empty["per_question"] = _zero_per_question(questions, answers, contexts, ground_truths)

    # RAGAS' default metrics use an LLM.  Avoid entering its retry loop when
    # the pipeline is intentionally being run without an API key (as in CI).
    if not OPENAI_API_KEY:
        return empty

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
            raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
        dataset = Dataset.from_dict({
            "question": list(questions), "answer": list(answers),
            "contexts": list(contexts), "ground_truth": list(ground_truths),
        })
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
        )
        embeddings = OpenAIEmbeddings(
            model=LLM_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=LLM_BASE_URL,
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        frame = result.to_pandas()
        per_question = []
        for index, row in frame.iterrows():
            try:
                source_index = int(index)
            except (TypeError, ValueError):
                source_index = len(per_question)
            if source_index < 0 or source_index >= len(questions):
                source_index = len(per_question)
            per_question.append(EvalResult(
                question=str(row.get("question", questions[source_index])),
                answer=str(row.get("answer", answers[source_index])),
                contexts=list(row.get("contexts", contexts[source_index]) or []),
                ground_truth=str(row.get("ground_truth", ground_truths[source_index])),
                **{name: _safe_score(row.get(name, 0.0)) for name in metric_names},
            ))
        # Some mocked/custom RAGAS results omit the row fields.  Keep the
        # report aligned with the input even in that case.
        if not per_question and questions:
            per_question = empty["per_question"]
        aggregate = {
            name: _mean(getattr(item, name) for item in per_question)
            for name in metric_names
        }
        return {**aggregate, "per_question": per_question}
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Implementation outline retained for lab review:
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    if not eval_results or bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    metric_names = tuple(diagnostic_tree)
    scored = []
    for result in eval_results:
        values = {name: _safe_score(_get_value(result, name, 0.0)) for name in metric_names}
        average = sum(values.values()) / len(values)
        # Stable metric priority makes ties reproducible and favors the order
        # used by the assignment's diagnostic tree.
        worst_metric = min(metric_names, key=lambda name: (values[name], metric_names.index(name)))
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": str(_get_value(result, "question", "")),
            "worst_metric": worst_metric,
            "score": values[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "average_score": average,
        })
    scored.sort(key=lambda item: (item["average_score"], item["score"]))
    for item in scored:
        item.pop("average_score", None)
    return scored[:bottom_n]


def _safe_score(value) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _get_value(item, name: str, default):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _zero_per_question(questions, answers, contexts, ground_truths) -> list[EvalResult]:
    count = min(len(questions), len(answers), len(contexts), len(ground_truths))
    return [EvalResult(str(questions[i]), str(answers[i]), list(contexts[i] or []),
                       str(ground_truths[i]), 0.0, 0.0, 0.0, 0.0)
            for i in range(count)]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
