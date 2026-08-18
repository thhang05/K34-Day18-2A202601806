from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, re, sys, json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GOOGLE_API_KEY, LLM_BASE_URL, LLM_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    # Implementation outline retained for lab review:
    # if GOOGLE_API_KEY:
    #     try:
    #         response = generate_text(
    #             messages=[
    #                 {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return resp.choices[0].message.content.strip()
    #     except Exception as e:
    #         print(f"  ⚠️  Google summarize failed: {e}")
    #
    # Extractive fallback (không cần API):
    # sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    # return ". ".join(sentences[:2]) + "." if sentences else text
    if not text or not text.strip():
        return ""
    if GOOGLE_API_KEY:
        try:
            response = _google_text(
                "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                text,
                max_tokens=150,
            )
            if response:
                return response
        except Exception as e:
            print(f"  ⚠️  Google summarize failed: {e}")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if s.strip()]
    if not sentences:
        return text.strip()
    summary = " ".join(sentences[:2]).strip()
    return summary if summary.endswith((".", "!", "?", "。", "！", "？")) else summary + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    # Implementation outline retained for lab review:
    # if GOOGLE_API_KEY:
    #     try:
    #         response = generate_text(
    #             messages=[
    #                 {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=200,
    #         )
    #         questions = resp.choices[0].message.content.strip().split("\n")
    #         return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
    #     except Exception as e:
    #         print(f"  ⚠️  Google HyQA failed: {e}")
    #
    # Extractive fallback:
    # import re
    # sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    if n_questions <= 0 or not text or not text.strip():
        return []
    if GOOGLE_API_KEY:
        try:
            response = _google_text(
                f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                "Trả về mỗi câu hỏi trên một dòng, bằng tiếng Việt.",
                text,
                max_tokens=200,
            )
            questions = _clean_questions(response, n_questions)
            if questions:
                return questions
        except Exception as e:
            print(f"  ⚠️  Google HyQA failed: {e}")

    sentences = [s.strip() for s in re.split(r"[.!?。！？\n]+", text) if len(s.strip()) > 10]
    questions = [f"{sentence.rstrip(' .!?。！？')}?" for sentence in sentences[:n_questions]]
    if not questions:
        questions = [f"Đoạn văn này nói về điều gì?"]
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    # Implementation outline retained for lab review:
    # if GOOGLE_API_KEY:
    #     try:
    #         context = generate_text(
    #             messages=[
    #                 {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
    #                 {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=80,
    #         )
    #         context = resp.choices[0].message.content.strip()
    #         return f"{context}\n\n{text}"
    #     except Exception as e:
    #         print(f"  ⚠️  Google contextual failed: {e}")
    #
    # Simple fallback:
    # prefix = f"Trích từ {document_title}. " if document_title else ""
    if not text:
        return ""
    if GOOGLE_API_KEY:
        try:
            context = _google_text(
                "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                "Chỉ trả về 1 câu.",
                f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                max_tokens=80,
            )
            if context:
                return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  Google contextual failed: {e}")

    prefix = f"Trích từ {document_title}. " if document_title else "Ngữ cảnh tài liệu: "
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    # Implementation outline retained for lab review:
    # if GOOGLE_API_KEY:
    #     try:
    #         import json as _json
    #         response = generate_text(
    #             messages=[
    #                 {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  Google metadata failed: {e}")
    #
    if not text or not text.strip():
        return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    if GOOGLE_API_KEY:
        try:
            response = _google_text(
                'Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ với các trường: '
                '{"topic": "...", "entities": ["..."], '
                '"category": "policy|hr|it|finance", "language": "vi|en"}',
                text,
                max_tokens=150,
            )
            metadata = _parse_json(response)
            if isinstance(metadata, dict):
                return _normalise_metadata(metadata)
        except Exception as e:
            print(f"  ⚠️  Google metadata failed: {e}")

    lowered = text.casefold()
    category = "policy"
    if any(word in lowered for word in ("mật khẩu", "vpn", "mfa", "phần mềm", "it ")):
        category = "it"
    elif any(word in lowered for word in ("lương", "thu nhập", "tài chính", "chi phí")):
        category = "finance"
    elif any(word in lowered for word in ("nhân viên", "nghỉ phép", "tuyển dụng", "đào tạo")):
        category = "hr"
    entities = sorted(set(re.findall(r"\b(?:20\d{2}|\d+\s*(?:ngày|tháng|năm))\b", text, re.I)))
    first_sentence = re.split(r"[.!?。！？\n]", text.strip(), maxsplit=1)[0].strip()
    return {"topic": first_sentence[:120] or "general", "entities": entities,
            "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    # Implementation outline retained for lab review:
    # if GOOGLE_API_KEY:
    #     try:
    #         import json as _json
    #         response = generate_text(
    #             messages=[
    #                 {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
    # {
    #   "summary": "tóm tắt 2-3 câu",
    #   "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
    #   "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
    #   "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
    # }"""},
    #                 {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=400,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  Enrichment API failed: {e}")
    if GOOGLE_API_KEY:
        try:
            response = _google_text(
                """Phân tích đoạn văn và trả về JSON hợp lệ, không markdown:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": [], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
                f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                max_tokens=400,
            )
            result = _parse_json(response)
            if isinstance(result, dict):
                return {
                    "summary": str(result.get("summary", "")),
                    "questions": _clean_questions(result.get("questions", []), 3),
                    "context": str(result.get("context", "")),
                    "metadata": _normalise_metadata(result.get("metadata", {})),
                }
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Keep the combined mode useful without an API key as well.
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text, 3),
        "context": f"Đoạn văn thuộc tài liệu {source}" if source else "Đoạn văn cung cấp thông tin chính sách nội bộ",
        "metadata": extract_metadata(text),
    }


def _google_text(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=GOOGLE_API_KEY, base_url=LLM_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return str(response.choices[0].message.content or "").strip()


def _parse_json(value: str):
    if not value:
        return None
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        return json.loads(match.group(0)) if match else None


def _clean_questions(value, limit: int) -> list[str]:
    if isinstance(value, str):
        values = value.splitlines()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return []
    cleaned = []
    for question in values:
        question = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(question)).strip()
        if question:
            cleaned.append(question if question.endswith(("?", "？")) else question + "?")
    return cleaned[:max(limit, 0)]


def _normalise_metadata(metadata) -> dict:
    if not isinstance(metadata, dict):
        return {}
    result = dict(metadata)
    result.setdefault("topic", "general")
    result.setdefault("entities", [])
    result.setdefault("category", "policy")
    result.setdefault("language", "vi")
    if not isinstance(result["entities"], list):
        result["entities"] = [str(result["entities"])]
    return result


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
