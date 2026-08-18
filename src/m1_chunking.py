from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from functools import lru_cache
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  [warn] Skip {os.path.basename(path)}: pypdf is not installed.")
        return ""

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    # 1. from sentence_transformers import SentenceTransformer
    #    from numpy import dot
    #    from numpy.linalg import norm
    # 2. metadata = metadata or {}
    # 3. Split text thành sentences: re.split(r'(?<=[.!?])\s+|\n\n', text)
    # 4. model = SentenceTransformer("all-MiniLM-L6-v2")
    #    embeddings = model.encode(sentences)
    # 5. cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-9)
    # 6. Duyệt từ sentence[1]:
    #      - sim(embedding[i-1], embedding[i]) < threshold → tách chunk mới
    #      - else: gộp vào chunk hiện tại
    # 7. Return [Chunk(text=joined_group, metadata={..., "strategy": "semantic"})]
    if not text or not text.strip():
        return []
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    metadata = dict(metadata or {})
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text)
                 if s.strip()]
    if not sentences:
        return []
    embeddings = _semantic_embeddings(sentences)
    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if _cosine_similarity(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([])
        groups[-1].append(sentences[i])
    return [Chunk(" ".join(group),
                  {**metadata, "chunk_index": i, "strategy": "semantic"})
            for i, group in enumerate(groups)]


def _lexical_embedding(sentence: str) -> dict[str, float]:
    tokens = re.findall(r"\w+", sentence.casefold(), flags=re.UNICODE)
    return {token: tokens.count(token) for token in set(tokens)}


@lru_cache(maxsize=1)
def _load_semantic_model():
    from sentence_transformers import SentenceTransformer
    # Never make a chunking call unexpectedly block on a model download.
    # Set HF_HUB_OFFLINE=0 and remove this option if online loading is desired.
    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


def _semantic_embeddings(sentences: list[str]):
    try:
        model = _load_semantic_model()
        try:
            return model.encode(sentences, convert_to_numpy=True)
        except TypeError:
            # Also support lightweight test doubles and older ST versions.
            return model.encode(sentences)
    except Exception:
        return [_lexical_embedding(sentence) for sentence in sentences]


def _cosine_similarity(a, b) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        numerator = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        denominator = (sum(v * v for v in a.values()) ** 0.5 *
                      sum(v * v for v in b.values()) ** 0.5)
    else:
        import numpy as np
        a, b = np.asarray(a), np.asarray(b)
        numerator = float(np.dot(a, b))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return numerator / (denominator + 1e-9)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    # 1. metadata = metadata or {}
    # 2. Split text bằng "\n\n" → paragraphs
    # 3. Gộp paragraphs thành parent chunks (mỗi parent ≤ parent_size chars):
    #      pid = f"parent_{len(parents)}"
    #      parents.append(Chunk(text=..., metadata={..., "chunk_type": "parent", "parent_id": pid}))
    # 4. Mỗi parent → split thành children (mỗi child ≤ child_size chars):
    #      children.append(Chunk(text=..., metadata={..., "chunk_type": "child"}, parent_id=pid))
    # 5. return (parents, children)
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    if not text or not text.strip():
        return [], []
    metadata = dict(metadata or {})
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = []
    for paragraph in paragraphs:
        units.extend(_split_fixed(paragraph, parent_size))
    parents, children = [], []
    current = ""
    for unit in units:
        candidate = unit if not current else current + "\n\n" + unit
        if current and len(candidate) > parent_size:
            _append_parent(current, metadata, parents, children, child_size)
            current = unit
        else:
            current = candidate
    if current:
        _append_parent(current, metadata, parents, children, child_size)
    return parents, children


def _split_fixed(text: str, size: int) -> list[str]:
    return [text[i:i + size].strip() for i in range(0, len(text), size)
            if text[i:i + size].strip()]


def _append_parent(text: str, metadata: dict, parents: list[Chunk],
                   children: list[Chunk], child_size: int) -> None:
    pid = f"parent_{len(parents)}"
    parents.append(Chunk(text, {**metadata, "chunk_index": len(parents),
                                "chunk_type": "parent", "parent_id": pid,
                                "strategy": "hierarchical"}))
    for child_index, child_text in enumerate(_split_fixed(text, child_size)):
        children.append(Chunk(child_text, {**metadata, "chunk_index": child_index,
                                           "chunk_type": "child",
                                           "strategy": "hierarchical"},
                              parent_id=pid))


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    # 1. metadata = metadata or {}
    # 2. sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    # 3. Duyệt sections:
    #      - Nếu match header (^#{1,3}\s+): lưu header hiện tại, tạo chunk cho content trước đó
    #      - Else: gộp vào content hiện tại
    # 4. Return [Chunk(text=header+content, metadata={..., "section": header, "strategy": "structure"})]
    if not text or not text.strip():
        return []
    metadata = dict(metadata or {})
    sections, header, content = [], "", []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = re.match(r"^(#{1,3}\s+.+?)\s*$", line) if not in_fence else None
        if match:
            if header or any(item.strip() for item in content):
                sections.append((header, content))
            header, content = match.group(1), []
        else:
            content.append(line)
    if header or any(item.strip() for item in content):
        sections.append((header, content))
    chunks = []
    for i, (section, lines) in enumerate(sections):
        body = "\n".join(lines).strip()
        chunk_text = "\n".join(part for part in (section, body) if part).strip()
        if chunk_text:
            chunks.append(Chunk(chunk_text, {**metadata, "chunk_index": i,
                                              "section": section,
                                              "strategy": "structure"}))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
