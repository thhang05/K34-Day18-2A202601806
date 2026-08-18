# Failure Analysis - Lab 18 Production RAG

## RAGAS Scores

| Metric | Naive baseline | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.0000* | 0.7250 | +0.7250 |
| Answer relevancy | 0.0000* | 0.3943 | +0.3943 |
| Context precision | 0.0000* | 0.8583 | +0.8583 |
| Context recall | 0.0000* | 0.8083 | +0.8083 |

`*` Baseline RAGAS was not evaluable because the API evaluator was unavailable; zero is a fallback value, not a quality claim.

## Bottom-5 Failures

### 1. Purchase approval for a 55-million device
- Worst metric: faithfulness (0.0000)
- Diagnosis: the answer was not sufficiently grounded in the retrieved evidence.
- Error tree: output incorrect -> inspect context -> inspect query intent and approval threshold.
- Suggested fix: use a stricter answer prompt and rerank/filter chunks by procurement topic and amount.

### 2. Senior employee with 9 years of service
- Worst metric: faithfulness (0.0000)
- Diagnosis: multi-hop answer combines leave and salary facts and is vulnerable to unsupported details.
- Error tree: output incomplete -> context may contain separate policy sections -> decompose leave and salary sub-questions.
- Suggested fix: require a citation for each numeric claim.

### 3. Maximum Junior probation salary
- Worst metric: faithfulness (0.0000)
- Diagnosis: numeric lookup answer was not fully supported by the generated response.
- Error tree: output unsupported -> check salary table version -> filter by role and probation policy.
- Suggested fix: use salary metadata filters and an exact-range answer template.

### 4. Annual leave during probation
- Worst metric: faithfulness (0.0000)
- Diagnosis: the answer may confuse probation eligibility with the general annual-leave policy.
- Error tree: output ambiguous -> inspect HR context -> distinguish rule, exception, and effective date.
- Suggested fix: add version-aware retrieval and explicitly state policy scope.

### 5. Training reimbursement after leaving at month eight
- Worst metric: faithfulness (0.0000)
- Diagnosis: the answer requires multiple policy clauses and a repayment calculation.
- Error tree: output unsupported -> check training agreement context -> verify duration and repayment percentage.
- Suggested fix: retrieve the complete training parent chunk and show calculation steps.

## Additional Findings

Answer relevancy is the weakest aggregate metric at 0.3943. Context precision (0.8583) and recall (0.8083) are strong, so retrieval is working better than answer generation. The temporary-advance question also has answer relevancy 0.3895, confirming that the generation issue affects multiple domains.

## Case Study

Question: `Muon mua thiet bi tri gia 55 trieu can ai phe duyet?`

1. Output correct? No, faithfulness is 0.0.
2. Context correct? Check procurement policy and the 50-million approval boundary.
3. Query rewrite OK? Rewrite into `nguoi phe duyet mua sam tren 50 trieu` and `co can xac nhan phong CNTT khong`.
4. Fix: rerank procurement chunks, filter by category, then answer with the exact threshold and required approval.

With one more hour, prioritize answer prompt tightening, query decomposition, and numeric/version-aware metadata filters.
