from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
                    HF_CACHE_DIR, QDRANT_LOCAL_FALLBACK)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
    except Exception:
        # Keep lexical search usable when the optional NLP dependency is not
        # installed (or cannot load its model in an offline environment).
        segmented = text
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(str(chunk.get("text", ""))).split()
                              for chunk in self.documents]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked_indices[:top_k]:
            if scores[i] <= 0:
                continue
            document = self.documents[i]
            results.append(SearchResult(
                text=str(document.get("text", "")),
                score=float(scores[i]),
                metadata=dict(document.get("metadata") or {}),
                method="bm25",
            ))
        return results


class _SimpleBM25:
    """Small fallback used when rank_bm25 is not installed."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_tokens)
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / max(self.doc_count, 1)
        self.doc_freqs = Counter()
        for doc in corpus_tokens:
            self.doc_freqs.update(set(doc))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for doc in self.corpus_tokens:
            freqs = Counter(doc)
            doc_len = len(doc)
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                df = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                tf = freqs[token]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
        self._encoder = None
        self._using_local_fallback = False

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            model_source = _cached_sentence_transformer_path(EMBEDDING_MODEL) or EMBEDDING_MODEL
            kwargs = {}
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                kwargs["token"] = hf_token
            if model_source != EMBEDDING_MODEL or os.getenv("HF_HUB_OFFLINE") == "1":
                kwargs["local_files_only"] = True
            try:
                self._encoder = SentenceTransformer(
                    model_source,
                    cache_folder=HF_CACHE_DIR,
                    **kwargs,
                )
            except TypeError:
                kwargs.pop("token", None)
                self._encoder = SentenceTransformer(model_source, **kwargs)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        try:
            self._recreate_collection(collection, VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        except Exception:
            if not QDRANT_LOCAL_FALLBACK:
                raise RuntimeError(
                    f"Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. "
                    "Start it with `docker compose up -d`, then run the pipeline again."
                ) from None
            from qdrant_client import QdrantClient
            print(
                f"  [warn] Qdrant at {QDRANT_HOST}:{QDRANT_PORT} unavailable; using in-memory Qdrant fallback.",
                flush=True,
            )
            self.client = QdrantClient(":memory:")
            self._using_local_fallback = True
            self._recreate_collection(collection, VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        if not chunks:
            return
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        encoder = self._get_encoder()
        try:
            vectors = encoder.encode(texts, show_progress_bar=True)
        except TypeError:
            vectors = encoder.encode(texts)
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            vector_values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            payload = {**(chunk.get("metadata") or {}), "text": texts[i]}
            points.append(PointStruct(id=i, vector=vector_values, payload=payload))
        self.client.upsert(collection, points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        vector = self._get_encoder().encode(query)
        query_vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        try:
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot query Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. "
                "Start it with `docker compose up -d`, then run the pipeline again."
            ) from None
        results = []
        for point in getattr(response, "points", response):
            payload = dict(getattr(point, "payload", None) or {})
            text = str(payload.get("text", ""))
            results.append(SearchResult(text=text, score=float(point.score),
                                        metadata=payload, method="dense"))
        return results

    def _recreate_collection(self, collection: str, vectors_config) -> None:
        if hasattr(self.client, "collection_exists") and self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=vectors_config)


def _cached_sentence_transformer_path(model_name: str) -> str | None:
    model_dir = "models--" + model_name.replace("/", "--")
    snapshots = Path(HF_CACHE_DIR, "hub", model_dir, "snapshots")
    if not snapshots.exists():
        return None
    for snapshot in sorted(snapshots.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if (
            snapshot.is_dir()
            and (snapshot / "modules.json").exists()
            and (snapshot / "config.json").exists()
            and ((snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists())
        ):
            return str(snapshot)
    return None


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    fused = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return [SearchResult(text=item["result"].text, score=float(item["score"]),
                         metadata=dict(item["result"].metadata), method="hybrid")
            for item in ranked[:top_k]]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
