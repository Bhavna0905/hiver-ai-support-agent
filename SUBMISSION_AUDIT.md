
# Submission Audit: Hiver AI Support Agent

Repository: `Bhavna0905/hiver-ai-support-agent`

Evaluation Target: `AmazonHelp` customer-support conversations from the Customer Support on Twitter dataset

Audit Purpose: Final pre-submission engineering and evaluation checklist

---

## 1. Executive Audit Matrix

| Dimension | Status | Evidence |
|---|---|---|
| Dataset reconstruction & brand selection | PASS | AmazonHelp selected from the Customer Support on Twitter dataset. Conversation relationships are reconstructed using `response_tweet_id` and `in_response_to_tweet_id`. |
| Grounded intent taxonomy | PASS | 13 intent categories were derived from recurring support patterns in the AmazonHelp data. |
| Intent classification | PASS | Hybrid MiniLM + TF-IDF representation with balanced Logistic Regression. |
| Historical resolution retrieval | PASS | MiniLM embeddings with FAISS retrieve historical customer → AmazonHelp resolution pairs. |
| Grounded response generation | PASS | Responses are generated from retrieved historical support examples using Groq/Qwen, with Ollama available as a local fallback. |
| Escalation routing | PASS | Deterministic safety and persistence rules decide whether to handle automatically or escalate. |
| Leakage prevention | PASS | Golden examples are excluded from classifier training, and conversations containing golden examples are excluded from the retrieval corpus. |
| Baseline comparison | PASS | Majority-class and TF-IDF + Logistic Regression baselines are evaluated against the main classifier. |
| Golden evaluation set | PASS | 200 finalized examples covering all 13 intents are used for evaluation. |
| Automated evaluation | PASS | Intent, escalation, retrieval, response-quality and end-to-end evaluation harnesses are implemented. |
| LLM-as-judge | PASS | 200 generated responses are evaluated across grounding, resolution correctness, relevance, safety and style. |
| Judge-human calibration | PASS | A 20-example subset was reviewed by one human reviewer and compared with the LLM judge. |
| Failure analysis | PASS | Recurring classification, retrieval, generation and escalation failure modes are documented. |
| Decision log | PASS | Non-obvious engineering decisions are documented in `reports/decision_log.md`. |
| Test suite | PASS | Deterministic tests cover core classifier, retrieval, escalation, generation, data and evaluation components. |
| Documentation | PASS | README and report document architecture, methodology, results, limitations and improvement plan. |

---

## 2. Dataset and Data Integrity

### Target Brand

The selected brand is:

```text
AmazonHelp
```

The project reconstructs conversational relationships from the Customer Support on Twitter dataset using:

```text
response_tweet_id
in_response_to_tweet_id
```

The complete raw dataset is approximately 516 MB and is intentionally not committed to the repository.

### Processed Data

The project produces:

- AmazonHelp conversation threads
- customer → AmazonHelp resolution pairs
- processed intent-training data
- FAISS retrieval index
- 200-example golden evaluation set

---

## 3. Intent Taxonomy

The final taxonomy contains 13 categories:

```text
delivery_problem
order_management
returns_refunds
product_issue
payment_billing
amazon_pay
account_access
prime_membership
digital_content
product_information
technical_issue
support_followup
other
```

The taxonomy was derived from recurring support patterns in the selected AmazonHelp data.

The goal was to keep the taxonomy small enough to be operationally meaningful while covering the major support patterns present in the data.

---

## 4. Intent Classification

### Main Model

The final classifier combines:

```text
Customer message
       │
       ├── MiniLM semantic embedding
       │
       └── TF-IDF word features
                │
                ▼
       Feature concatenation
                │
                ▼
       Balanced Logistic Regression
                │
                ▼
          13-class intent
```

### Evaluation

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 18.5% | 0.024 |
| TF-IDF + Logistic Regression | 44.0% | 0.416 |
| **MiniLM + TF-IDF Hybrid** | **55.0%** | **0.541** |

The main classifier improves substantially over the lexical-only baseline.

The classifier is nevertheless not treated as production-ready because several intent boundaries remain difficult.

---

## 5. Retrieval

Historical customer-support resolutions are indexed using:

```text
all-MiniLM-L6-v2
        ↓
Normalized embeddings
        ↓
FAISS IndexFlatIP
```

The retrieval corpus excludes conversations containing golden evaluation examples.

### Retrieval Diagnostics

| Metric | Result |
|---|---:|
| Recall@1 | 46.0% |
| Recall@3 | 62.0% |
| Recall@5 | 66.0% |
| MRR | 0.537 |

These metrics are explicitly treated as approximate diagnostics because retrieval relevance is estimated using an evaluation-time intent lexicon rather than an independently human-labelled retrieval relevance dataset.

---

## 6. Response Generation

The primary response generator uses:

```text
Groq
qwen/qwen3.8-27b
```

A local fallback is available through:

```text
Ollama
llama3.2:3b
```

The response generator uses retrieved historical examples as grounding evidence.

Response quality is evaluated across:

1. Grounding
2. Resolution correctness
3. Relevance
4. Safety
5. Style

### LLM Judge Results

| Dimension | Mean / 5 |
|---|---:|
| Grounding | 3.85 |
| Resolution correctness | 3.06 |
| Relevance | 3.42 |
| Safety | 4.88 |
| Style | 4.44 |
| Overall | 3.88 |

The main observed weakness is exact historical-resolution transfer.

---

## 7. Escalation

The escalation component uses deterministic safety and persistence rules.

The system considers signals such as:

- security/account compromise
- financial-risk situations
- repeated support contact
- unresolved issues
- requests requiring human intervention

### Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 87.5% |
| Precision | 66.7% |
| Recall | 61.1% |
| F1 | 0.638 |
| False auto-handle | 7.0% |
| False escalation | 5.5% |

False auto-handle is tracked separately because incorrectly handling a case automatically can be more costly than unnecessarily escalating a routine case.

---

## 8. Golden Evaluation Set

The final evaluation benchmark contains:

```text
200 examples
13 intent categories
```

Each example includes:

- customer message
- expected intent
- expected escalation decision
- annotation metadata
- annotation notes

The finalized dataset was human-reviewed.

The reviewed labels are treated as the benchmark ground truth.

---

## 9. Leakage Prevention

The project explicitly separates evaluation data from inference data.

The main controls are:

| Control | Purpose |
|---|---|
| Golden holdout | Golden examples excluded from classifier training |
| Conversation holdout | Conversations containing golden examples excluded from retrieval |
| Retrieval self-exclusion | Evaluation examples cannot retrieve themselves |
| Gold-label isolation | Expected labels are used only during evaluation |
| Deduplication | Duplicate training examples are removed |

The strongest retrieval leakage control is conversation-level exclusion rather than only excluding individual tweet IDs.

---

## 10. Evaluation Harness

The project provides independent and end-to-end evaluation commands.

### Main End-to-End Evaluation

```powershell
python -m evaluation.end_to_end_eval
```

### Additional Evaluation

```powershell
python -m evaluation.evaluate
python -m evaluation.retrieval_metrics
python -m evaluation.judge
python -m evaluation.human_agreement
```

Evaluation artifacts are written to:

```text
reports/
├── end_to_end_results.json
├── retrieval_results.json
├── llm_judge_results.json
└── human_agreement.json
```

---

## 11. LLM Judge Calibration

A 200-example subset was additionally reviewed by one human reviewer.

The comparison produced:

| Metric | Result |
|---|---:|
| Exact agreement | 45.0% |
| Quadratic weighted κ | 0.465 |

Important limitation:

> The calibration uses one human reviewer rather than multiple independent raters.

Therefore the LLM judge is treated as a secondary evaluation signal, not as human ground truth.

---

## 12. Failure Analysis

The main recurring failure modes identified during evaluation are:

### 1. Ambiguous Intent Boundaries

Several support categories overlap, particularly:

- delivery
- order management
- product issues
- returns/refunds

### 2. Missing Conversation Context

Short Twitter messages often do not contain enough information to identify the customer's actual problem.

### 3. Digital-Content Prediction Bias

`digital_content` has high recall but low precision:

```text
Precision: 30.6%
Recall:    91.7%
```

The classifier therefore tends to over-predict this category.

### 4. Incorrect Resolution Transfer

A semantically similar historical example does not necessarily imply that the same resolution is correct for the current customer.

### 5. Escalation and Unusual-Phrasing Errors

Repeated-contact signals, malformed messages and unusual/non-English phrasing can create difficult routing and classification cases.

Detailed failure analysis is documented in:

```text
reports/report.md
```

---

## 13. Confidence Analysis

Classifier confidence provides useful but incomplete information.

| Confidence | Examples | Accuracy |
|---|---:|---:|
| `< 0.50` | 75 | 45.3% |
| `0.50–0.70` | 39 | 43.6% |
| `0.70–0.85` | 29 | 51.7% |
| `>= 0.85` | 57 | 77.2% |

This indicates that high-confidence predictions are considerably more reliable.

However, confidence alone is not used as the complete escalation policy.

---

## 14. Reproducibility

The project provides:

- `requirements.txt`
- `.env.example`
- deterministic random seeds where applicable
- evaluation scripts
- saved model artifacts
- saved retrieval artifacts
- evaluation result artifacts
- documented commands
- deterministic tests

The README documents the setup and evaluation workflow.

The raw dataset is intentionally excluded from Git because of its size.

---

## 15. Test Suite

Run:

```powershell
python -m pytest -q
```

The test suite covers core functionality including:

- intent classification
- retrieval
- escalation rules
- response generation
- data processing
- evaluation metrics

---

## 16. Documentation

The repository contains:

```text
README.md
reports/report.md
reports/decision_log.md
```

### README

Documents:

- architecture
- setup
- evaluation
- results
- methodology
- failure analysis
- limitations
- one-week improvement plan

### Report

Contains detailed failure analysis and supporting discussion.

### Decision Log

Documents non-obvious engineering decisions and their rationale.

---

## 17. What Is Not Claimed

This project intentionally does not claim:

- production readiness
- perfect intent classification
- production-grade multi-turn dialogue management
- independent multi-rater annotation agreement
- independently human-labelled retrieval relevance
- a production vector database
- live Twitter integration

The evaluation is intended to demonstrate a reproducible prototype and make its limitations measurable.

---

## 18. Headline Results

The main current benchmark results are:

```text
Intent Accuracy:       55.0%
Intent Macro F1:       0.541

Escalation Accuracy:   87.5%
Escalation F1:         0.638

Retrieval Recall@5:    66.0%
Retrieval MRR:         0.537

LLM Judge Overall:     3.88 / 5
```

The strongest observed areas are:

- safety
- response style
- escalation routing

The largest opportunities are:

- intent disambiguation
- conversation context
- exact historical-resolution transfer

---

## 19. Final Submission Checklist

Before submission, verify:

- [ ] README contains current benchmark numbers
- [ ] `reports/report.md` is present
- [ ] `reports/decision_log.md` is present
- [ ] 200-example golden set is present
- [ ] `.env` is not tracked
- [ ] API keys/secrets are not committed
- [ ] `requirements.txt` is present
- [ ] tests pass
- [ ] evaluation artifacts are present where intended
- [ ] GitHub repository is accessible to the evaluator
- [ ] Repository contains no accidental large raw dataset
- [ ] README commands match the actual repository
- [ ] Final Git commit has been pushed successfully

---

## 20. Final Status

**SUBMISSION READY — pending final local verification**

The project contains the required support-agent pipeline, evaluation harness, golden benchmark, leakage controls, LLM-judge analysis, failure analysis and engineering decision log.

The project intentionally reports both strengths and limitations rather than presenting a single benchmark number as proof of production readiness.

Final verification should be performed immediately before submission to ensure that the repository state, documentation, tests and evaluation artifacts are synchronized.