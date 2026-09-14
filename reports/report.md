# Hiver AI Support Agent — Evaluation Report

## 1. Problem Framing

The goal is to build an AI customer-support agent that can:

1. classify an incoming customer message into a small set of support intents;
2. retrieve historical examples showing how similar issues were resolved;
3. draft a grounded response based on those historical resolutions; and
4. decide whether the issue can be handled automatically or should be escalated to a human.

The system was built for the AmazonHelp brand from the Customer Support on Twitter dataset.

The pipeline is:

Customer message  
→ Intent classification  
→ Historical-resolution retrieval  
→ Response generation  
→ Escalation policy  
→ Final response + decision + reason

The evaluation uses a 200-example golden dataset that was reviewed by a human and is kept separate from training and retrieval data.

---

## 2. Approach

### Intent Classification

The final classifier uses `all-MiniLM-L6-v2` sentence embeddings followed by balanced logistic regression.

The taxonomy contains 13 intents:

- `delivery_problem`
- `order_management`
- `returns_refunds`
- `product_issue`
- `payment_billing`
- `amazon_pay`
- `account_access`
- `prime_membership`
- `digital_content`
- `product_information`
- `technical_issue`
- `support_followup`
- `other`

An explicit `other` class prevents messages that do not represent an operational support issue from being forced into an unrelated category.

### Historical Resolution Retrieval

Historical customer → AmazonHelp resolution pairs are embedded using MiniLM and indexed with FAISS.

The retrieval corpus excludes conversations containing golden evaluation examples to reduce evaluation leakage.

### Response Generation

The primary response generator uses Groq with `qwen/qwen3.8-27b`. A local Llama 3.2 3B model running through Ollama is used as a fallback when the primary provider is unavailable.

The generator is prompted to use retrieved historical resolutions as evidence and avoid inventing policies, actions, dates, refunds, or tracking information.

### Escalation

Escalation is deliberately separated from intent classification.

The policy considers:

- security and fraud signals;
- repeated unsuccessful contact;
- repeated attempts;
- persistent unresolved issues;
- insufficient confidence/evidence;
- sensitive support requests.

This avoids blanket rules such as escalating every account-related issue.

---

## 3. Baselines and Results

Evaluation was performed on 200 golden examples.

### Intent Classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 15.0% | 0.020 |
| TF-IDF + Logistic Regression | 38.5% | 0.366 |
| MiniLM + Logistic Regression | **48.5%** | **0.478** |

The semantic classifier improves substantially over both the trivial and simple lexical baselines.

However, 48.5% accuracy also shows that intent boundaries in this dataset are difficult and that the classifier remains the main system weakness.

### Escalation

| Metric | Final system |
|---|---:|
| Accuracy | **87.5%** |
| Precision | **66.7%** |
| Recall | **61.1%** |
| F1 | **0.638** |
| False auto-handle rate | **7.0%** |
| False escalation rate | **5.5%** |

The escalation policy was evaluated separately and then as part of the complete pipeline.

---

## 4. Retrieval Results

On the 200-example evaluation set:

| Metric | Result |
|---|---:|
| Recall@1 | 46.0% |
| Recall@3 | 62.0% |
| Recall@5 | 66.0% |
| MRR | 0.537 |

These retrieval metrics use an evaluation-time intent lexicon to estimate whether retrieved examples belong to the same broad intent.

Therefore, they should be interpreted as approximate retrieval diagnostics rather than definitive semantic-grounding ground truth.

---

## 5. Response Quality

Generated responses were evaluated using a local LLM-as-a-judge (`llama3.2:3b`) across all 200 examples.

| Dimension | Score / 5 |
|---|---:|
| Grounding | **3.85** |
| Resolution correctness | **3.06** |
| Relevance | **3.42** |
| Safety | **4.88** |
| Style | **4.44** |
| **Overall** | **3.88** |

The strongest properties are safety and style, while resolution correctness remains the main weakness.

The agent generally produces safe, relevant responses grounded in historical examples, but it does not always transfer the exact historical resolution to the current customer's situation.

### Judge-Human Agreement

The judge was calibrated against human ratings on a 20-example subset.

| Dimension | Exact Agreement | Quadratic Weighted κ |
|---|---:|---:|
| Grounding | 15.0% | 0.148 |
| Resolution correctness | 35.0% | 0.354 |
| Relevance | 10.0% | 0.005 |
| Safety | 35.0% | -0.136 |
| Style | 35.0% | 0.161 |
| **Overall** | **45.0%** | **0.465** |

Agreement is moderate overall but weak for several individual dimensions.

Therefore, LLM-judge scores are treated as a secondary evaluation signal rather than ground truth. The calibration used one human reviewer.

---

## 6. Top Five Failure Modes

### 1. Ambiguous Intent Boundaries

Several messages legitimately contain signals for multiple intents.

For example, a message about a damaged product may also mention delivery or replacement. The classifier can therefore select `returns_refunds`, `delivery_problem`, or `product_issue` depending on wording.

**Hypothesis:** The taxonomy contains overlapping operational concepts, and short Twitter messages provide insufficient context.

**Potential improvement:** Hierarchical classification or multi-label candidate generation followed by a second-stage intent decision.

### 2. Low-Context and Malformed Messages

Some tweets are extremely short, multilingual, or corrupted by encoding issues.

For example, tweet `1668372` contains heavily corrupted text and was classified as `digital_content` while the gold label is `other`.

**Hypothesis:** Embedding-based classification cannot reliably infer an operational intent when the text itself does not contain enough usable information.

**Potential improvement:** Detect unsupported or malformed input and route it to `other` or human review.

### 3. Confidence Is Not Sufficient for Escalation

Some incorrect classifications receive relatively high confidence.

For example, tweet `1743707` is a product-information question but was classified as `product_issue` with 0.955 confidence.

This demonstrates that model confidence is not equivalent to correctness.

**Hypothesis:** Logistic-regression probability is useful for ranking uncertainty but is not a calibrated estimate of correctness across all intents.

**Potential improvement:** Confidence calibration and class-specific thresholds.

### 4. Escalation Signals Can Overlap With Normal Support Requests

Persistence words such as "still waiting", "refund", or "already informed" can occur in ordinary cases that do not require human escalation.

Examples include cases where a customer is simply asking for an expected refund or delivery update.

**Hypothesis:** Lexical escalation rules are useful safety guards but can over-trigger when context is missing.

**Potential improvement:** Combine persistence signals with severity, failed-resolution evidence, and retrieved historical outcomes.

### 5. Safe Responses Are Not Always Resolving Responses

The LLM judge gives a high safety score of **4.88/5** but a lower resolution-correctness score of **3.06/5**.

This means the system generally avoids unsafe responses but does not always provide the concrete action needed to solve the customer's problem.

**Hypothesis:** Historical retrieval provides evidence of similar resolutions, but the response model can fail to correctly transfer that resolution to the current situation.

**Potential improvement:** Structured response generation with explicit evidence extraction and a post-generation resolution check.

---

## 7. What Is Misleading About My Headline Number?

The headline intent accuracy of **48.5%** is useful but incomplete.

First, the test set contains only 200 examples, so the number has sampling uncertainty.

Second, the intent taxonomy contains overlapping categories. A prediction can be operationally reasonable while still being counted as incorrect because it differs from the single gold label.

Third, the retrieval evaluation uses an approximate intent lexicon rather than independently annotated retrieval relevance.

Fourth, the LLM judge is itself an imperfect evaluator. Its scores should therefore be treated as supporting evidence rather than absolute truth.

Finally, aggregate accuracy hides per-intent weaknesses and difficult examples.

For these reasons, the more honest headline is:

> The system demonstrates a complete, leakage-controlled support-agent pipeline that substantially beats simple intent baselines, achieves 0.638 escalation F1 on 200 reviewed examples, and produces generally safe responses, but intent classification and exact resolution correctness remain important limitations.

---

## 8. Key Findings

- MiniLM intent classification reaches **48.5% accuracy** and **0.478 macro F1**, outperforming both the majority-class and TF-IDF baselines.
- The escalation policy reaches **0.638 F1**, with a **7.0% false auto-handle rate** and **5.5% false escalation rate**.
- Retrieval reaches **66.0% Recall@5**, although the retrieval evaluation uses an approximate intent-based relevance signal.
- Generated responses score **4.88/5 for safety**, **4.44/5 for style**, and **3.88/5 overall**.
- The main remaining weaknesses are ambiguous intent boundaries, low-context/malformed messages, imperfect retrieval, and responses that are safe but not always sufficiently resolving.

---

## 9. One-Week Improvement Plan

If given another week, I would:

1. Calibrate classifier confidence and add a second-stage classifier for ambiguous intent pairs.
2. Improve malformed and low-context message detection and route uncertain cases to `other` or human review.
3. Extract structured resolution information from retrieved historical examples before generation.
4. Add a post-generation check for resolution completeness and contradictions with retrieved evidence.
5. Expand human evaluation beyond the current 20-example judge-calibration subset.