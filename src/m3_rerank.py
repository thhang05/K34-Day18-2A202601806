from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, re, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_CACHE_DIR, RERANK_TOP_K, USE_CROSS_ENCODER


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_error: Exception | None = None

    def _load_model(self):
        if self._model is None:
            if self._load_error is not None:
                return None
            if not USE_CROSS_ENCODER:
                return None
            try:
                # CrossEncoder is deliberately imported lazily: importing this
                # module must remain possible in lightweight/offline setups.
                from sentence_transformers import CrossEncoder
                model_source = _cached_transformer_path(self.model_name) or self.model_name
                try:
                    # Avoid an unbounded Hugging Face network retry in CI or
                    # an offline workstation when weights are not cached.
                    self._model = CrossEncoder(model_source, local_files_only=True)
                except TypeError:
                    # Compatibility with older sentence-transformers and small
                    # test doubles that only accept the model name.
                    self._model = CrossEncoder(model_source)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                # Retrieval should still be usable if the optional model or its
                # weights are unavailable.  The lexical fallback below keeps
                # the pipeline deterministic and is also useful for tests.
                self._load_error = exc
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []

        # Do not mutate the search result dictionaries.  This matters because
        # the same candidates can be passed to more than one reranker.
        candidates = [doc for doc in documents if isinstance(doc, dict)]
        if not candidates:
            return []
        model = self._load_model()
        if model is not None:
            pairs = [(str(query), str(doc.get("text", ""))) for doc in candidates]
            try:
                scores = model.predict(pairs)
                scores = _normalise_scores(scores, len(candidates))
            except (TypeError, ValueError, RuntimeError):
                scores = _lexical_scores(query, candidates)
        else:
            scores = _lexical_scores(query, candidates)

        scored = sorted(
            zip(scores, candidates), key=lambda item: float(item[0]), reverse=True
        )
        return [
            RerankResult(
                text=str(doc.get("text", "")),
                original_score=float(doc.get("score", 0.0) or 0.0),
                rerank_score=float(score),
                metadata=dict(doc.get("metadata") or {}),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored[:top_k], start=1)
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None
        self._load_error: Exception | None = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            if self._model is None and self._load_error is None:
                from flashrank import Ranker
                self._model = Ranker()
            from flashrank import RerankRequest
            passages = [{"text": str(doc.get("text", "")), "_doc": doc} for doc in documents]
            ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
            output = []
            for rank, item in enumerate(ranked[:top_k], start=1):
                doc = item.get("_doc", item)
                output.append(RerankResult(
                    text=str(doc.get("text", "")),
                    original_score=float(doc.get("score", 0.0) or 0.0),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=dict(doc.get("metadata") or {}), rank=rank,
                ))
            return output
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._load_error = exc
            return _results_from_scores(_lexical_scores(query, documents), documents, top_k)


def _normalise_scores(scores: Any, expected: int) -> list[float]:
    """Convert numpy/tensor/scalar CrossEncoder output to plain floats."""
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        scores = [scores]
    scores = list(scores)
    if len(scores) != expected:
        raise ValueError("CrossEncoder returned a score for the wrong number of documents")
    return [float(score) for score in scores]


def _cached_transformer_path(model_name: str) -> str | None:
    model_dir = "models--" + model_name.replace("/", "--")
    snapshots = Path(HF_CACHE_DIR, "hub", model_dir, "snapshots")
    if not snapshots.exists():
        return None
    for snapshot in sorted(snapshots.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if (
            snapshot.is_dir()
            and (snapshot / "config.json").exists()
            and ((snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists())
        ):
            return str(snapshot)
    return None


def _lexical_scores(query: str, documents: list[dict]) -> list[float]:
    """Small offline fallback based on token overlap and the original score."""
    query_tokens = set(re.findall(r"\w+", str(query).casefold()))
    scores = []
    for doc in documents:
        text = str(doc.get("text", ""))
        tokens = set(re.findall(r"\w+", text.casefold()))
        overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
        phrase_bonus = 1.0 if str(query).casefold() in text.casefold() else 0.0
        original = float(doc.get("score", 0.0) or 0.0)
        scores.append(overlap + phrase_bonus + 1e-6 * original)
    return scores


def _results_from_scores(scores: list[float], documents: list[dict], top_k: int) -> list[RerankResult]:
    scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
    return [RerankResult(str(doc.get("text", "")), float(doc.get("score", 0.0) or 0.0),
                         float(score), dict(doc.get("metadata") or {}), rank)
            for rank, (score, doc) in enumerate(scored[:top_k], start=1)]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
