# Individual Reflection - Lab 18

**Name:** Hồ Thúy Hằng
**Modules:** M1, M2, M3 , M4,  M5

==============================================================================================================

PRODUCTION RAG SCORES

============================================================

  ✗ faithfulness: 0.7250

  ✗ answer_relevancy: 0.3943

  ✓ context_precision: 0.8583

  ✓ context_recall: 0.8083

Report saved to ragas_report.json==========

PRODUCTION RAG SCORES

============================================================

  ✗ faithfulness: 0.7250

  ✗ answer_relevancy: 0.3943

  ✓ context_precision: 0.8583

  ✓ context_recall: 0.8083

Report saved to ragas_report.jsoM4, M5

## 1. Technical contribution

Implemented semantic, hierarchical, and structure-aware chunking; Vietnamese BM25 plus dense search and RRF; CrossEncoder-compatible reranking with an offline fallback; RAGAS evaluation with four metrics and diagnostic failure analysis; and combined M5 enrichment with local fallback.

Main functions: `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware`, `reciprocal_rank_fusion`, `CrossEncoderReranker.rerank`, `evaluate_ragas`, `failure_analysis`, and `enrich_chunks`.

All automated tests pass: 37/37.

## 2. Knowledge learned

RAG quality has separate retrieval and generation stages. In this run, context precision and recall were above 0.80, while answer relevancy was only 0.3943. RRF combines keyword and semantic ranking without requiring score normalization. Hierarchical chunks preserve broader context while keeping child retrieval precise.

## 3. Difficulty and solution

The largest debugging issue was the NumPy array truth-value error while converting RAGAS rows. The expression using `value or []` was invalid for an array. I replaced it with explicit context normalization and verified the fix with the M4 tests. The pipeline also needed an offline fallback when OpenAI connectivity was unavailable.

## 4. If I did it again

I would add version-aware metadata filters for policies, decompose multi-hop questions, and require every numeric answer to be supported by an evidence span. I would also make report paths consistent between the root assignment format and the `reports/` directory.

## 5. Self-assessment

| Criterion                | Score (1-5) |
| ------------------------ | ----------: |
| Understanding of lecture |           4 |
| Code quality             |           4 |
| Teamwork                 |           4 |
| Problem solving          |           5 |
