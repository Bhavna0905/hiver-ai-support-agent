# Hiver AI Support Agent — classify, draft, escalate

An AI customer-support agent for `@AmazonHelp` built from the **Customer Support on Twitter** dataset.

The system classifies incoming customer messages, retrieves similar historical resolutions, drafts a grounded response, and decides whether the request should be handled automatically or escalated to a human.

---

## 1. What I Built

The project consists of four main components:

1. **Intent classification** — classifies customer messages into a 13-category taxonomy derived from the AmazonHelp data.
2. **Historical resolution retrieval** — retrieves similar customer → AmazonHelp resolution pairs using semantic similarity.
3. **Grounded reply generation** — generates a response using retrieved historical examples.
4. **Escalation routing** — determines whether the request should be automatically handled or escalated to a human.

The evaluation harness measures these components independently and end-to-end.

### What I Deliberately Did Not Build

- Live Twitter integration
- Production web UI
- Cloud-hosted infrastructure
- Fine-tuning on the 200-example golden set
- Full production-grade multi-turn dialogue management
- Production vector database

The focus was on building a reproducible support-agent prototype and evaluating its strengths and failure modes.

---

## 2. Architecture

```text
Customer Message
       │
       ▼
┌─────────────────────────────┐
│      Intent Classifier      │
│ MiniLM + TF-IDF + Logistic  │
│        Regression           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Historical Retrieval     │
│       MiniLM + FAISS        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Response Generator      │
│     Groq / Qwen 3.8 27B     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Escalation Policy      │
│ Safety + Persistence Rules  │
└─────────────┬───────────────┘
              │
              ▼
Final Response
+ Intent
+ Retrieved Evidence
+ Escalation Decision
+ Reason
```

### Components

| Component | Implementation |
|---|---|
| Intent classifier | `all-MiniLM-L6-v2` embeddings + TF-IDF word features + balanced Logistic Regression |
| Retrieval | MiniLM embeddings + FAISS |
| Response generation | Groq `qwen/qwen3.8-27b` |
| Local fallback | Ollama `llama3.2:3b` |
| Escalation | Deterministic safety and persistence rules |

### Intent Classification Pipeline

The main classifier combines semantic and lexical representations:

```text
Customer message
       │
       ├──────────────────┐
       ▼                  ▼
MiniLM embedding     TF-IDF features
       │                  │
       └────────┬─────────┘
                ▼
       Feature concatenation
                │
                ▼
      Balanced Logistic Regression
                │
                ▼
           13-class intent
```

This hybrid representation performed better on the golden evaluation set than the TF-IDF-only baseline.

---

## 3. Target Brand

The selected brand is **AmazonHelp**.

The raw **Customer Support on Twitter** dataset contains conversation relationships through fields such as:

- `response_tweet_id`
- `in_response_to_tweet_id`

These relationships are used to reconstruct customer → brand resolution pairs.

The full raw dataset is approximately **516 MB** and is not included in the repository.

---

## 4. Intent Taxonomy

The final taxonomy contains 13 intents:

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

The taxonomy was derived from recurring support patterns in the AmazonHelp data rather than importing an unrelated benchmark taxonomy.

---

## 5. Quickstart

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure Groq

Create a local `.env` file:

```powershell
Copy-Item .env.example .env
```

Add your API key:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b

OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434

LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512
LLM_TIMEOUT=120
```

`.env` is excluded from Git.

### Local fallback

If Groq is unavailable, Ollama can be used locally:

```powershell
ollama pull llama3.2
```

The response generator contains fallback logic so that the agent can continue using the local model when required.

---

## 6. Reproduce the Evaluation

The main end-to-end evaluation can be run with:

```powershell
python -m evaluation.end_to_end_eval
```

The evaluation uses the finalized **200-example golden set**.

Additional evaluation commands:

```powershell
python -m evaluation.evaluate
python -m evaluation.retrieval_metrics
python -m evaluation.judge
python -m evaluation.human_agreement
```

The resulting artifacts are written to:

```text
reports/
├── end_to_end_results.json
├── retrieval_results.json
├── llm_judge_results.json
└── human_agreement.json
```

---

## 7. Results

The evaluation uses **200 human-reviewed examples**.

Golden examples are excluded from classifier training and from the retrieval corpus to reduce evaluation leakage.

### Intent Classification

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 18.5% | 0.024 |
| TF-IDF + Logistic Regression | 44.0% | 0.416 |
| **MiniLM + TF-IDF Hybrid** | **55.0%** | **0.541** |

The hybrid semantic + lexical classifier improves substantially over the simple TF-IDF baseline.

However, short and ambiguous customer-support messages remain difficult to classify reliably.

### Confidence Analysis

The classifier's confidence is informative but not sufficient by itself for production routing.

| Confidence | Examples | Accuracy |
|---|---:|---:|
| `< 0.50` | 75 | 45.3% |
| `0.50–0.70` | 39 | 43.6% |
| `0.70–0.85` | 29 | 51.7% |
| `>= 0.85` | 57 | **77.2%** |

This suggests that high-confidence predictions are substantially more reliable, while low-confidence predictions remain challenging.

I therefore avoid treating a single confidence threshold as a complete escalation strategy.

### Escalation

| Metric | Result |
|---|---:|
| Accuracy | **87.5%** |
| Precision | **66.7%** |
| Recall | **61.1%** |
| F1 | **0.638** |
| False auto-handle | **7.0%** |
| False escalation | **5.5%** |

False auto-handle is tracked separately because automatically handling a message that should have gone to a human is more operationally important than a harmless extra escalation.

### Retrieval

| Metric | Result |
|---|---:|
| Recall@1 | 46.0% |
| Recall@3 | 62.0% |
| Recall@5 | **66.0%** |
| MRR | **0.537** |

These retrieval metrics are approximate diagnostics because retrieved intent relevance is estimated using an evaluation-time intent lexicon rather than independently human-labelled retrieval relevance.

### Response Quality

Responses were evaluated across all 200 examples using a local `llama3.2:3b` LLM judge.

| Dimension | Score / 5 |
|---|---:|
| Grounding | **3.85** |
| Resolution correctness | **3.06** |
| Relevance | **3.42** |
| Safety | **4.88** |
| Style | **4.44** |
| **Overall** | **3.88** |

The strongest dimensions are safety and style.

The main weakness is **exact resolution correctness**: a response can be relevant and well-written while still failing to reproduce the precise resolution that historically solved the customer's problem.

---

## 8. Evaluation Methodology

### Baselines

Two baselines are used to provide context for the main classifier.

#### Majority-Class Baseline

Always predicts the most frequent intent in the training data.

This establishes the accuracy available simply from the class distribution.

#### TF-IDF Baseline

Uses:

```text
TF-IDF word n-grams
        ↓
Logistic Regression
        ↓
Intent prediction
```

The main system adds semantic MiniLM features to the lexical representation.

### Main Classifier

The main intent classifier uses:

```text
Customer message
       │
       ├── MiniLM semantic embedding
       │
       └── TF-IDF word n-grams
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

Training data is deduplicated and excludes the golden evaluation examples.

### Metrics

#### Intent Classification

- Accuracy
- Macro F1
- Weighted F1
- Per-intent precision
- Per-intent recall
- Per-intent F1

#### Escalation

- Accuracy
- Precision
- Recall
- F1
- False auto-handle rate
- False escalation rate

#### Retrieval

- Recall@1
- Recall@3
- Recall@5
- MRR

#### Response Quality

- Grounding
- Resolution correctness
- Relevance
- Safety
- Style

---

## 9. Golden Set

The evaluation set contains **200 examples** covering the complete 13-intent taxonomy.

Each example contains:

- Customer message
- Expected intent
- Expected escalation decision
- Annotation metadata
- Annotation notes

The finalized dataset was human-reviewed.

The evaluation pipeline treats the reviewed labels as the benchmark ground truth.

### Leakage Prevention

Golden examples are excluded from:

1. Classifier training
2. Retrieval candidates

Additionally, conversations containing golden examples are excluded from the retrieval corpus.

This prevents the system from simply retrieving the exact evaluation example during evaluation.

---

## 10. Leakage Controls

The project explicitly separates evaluation data from inference data.

The main controls are:

| Control | Purpose |
|---|---|
| Golden holdout | Golden examples excluded from classifier training |
| Conversation holdout | Conversations containing golden examples excluded from retrieval |
| Retrieval self-exclusion | Evaluated examples cannot retrieve themselves |
| Gold-label isolation | Expected labels are used only for evaluation |
| Deduplication | Duplicate training examples are removed |

The goal is to make the reported metrics represent actual generalization rather than memorization.

---

## 11. LLM-as-Judge

Response quality is evaluated using a local Ollama model:

```text
Judge model:
llama3.2:3b
```

The judge evaluates five dimensions on a 1–5 scale:

1. Grounding
2. Resolution correctness
3. Relevance
4. Safety
5. Style

The judge is used as a secondary evaluation signal rather than being treated as ground truth.

### Judge Results

| Dimension | Mean |
|---|---:|
| Grounding | 3.85 |
| Resolution correctness | 3.06 |
| Relevance | 3.42 |
| Safety | 4.88 |
| Style | 4.44 |
| Overall | 3.88 |

---

## 12. Judge–Human Calibration

A 20-example subset was also rated by a human reviewer and compared against the LLM judge.

| Metric | Overall |
|---|---:|
| Exact agreement | **45.0%** |
| Quadratic weighted κ | **0.465** |

The calibration uses **one human reviewer**.

Agreement is imperfect, particularly for subjective dimensions, so LLM-judge scores should not be interpreted as equivalent to human ground truth.

The calibration artifact is stored in:

```text
reports/human_agreement.json
```

---

## 13. Failure Analysis

The evaluation revealed several recurring failure modes.

### 1. Ambiguous Intent Boundaries

Some customer messages can reasonably belong to multiple operational categories.

For example, order-related messages may overlap with:

- delivery problems
- product issues
- returns/refunds
- order management

This causes confusion between closely related intents.

### 2. Short or Context-Poor Messages

Twitter support messages are often extremely short.

Without surrounding conversation context, a message may not contain enough information to determine the correct intent.

### 3. Digital-Content Prediction Bias

`digital_content` is frequently predicted for messages whose actual intent belongs to another category.

This is reflected in the evaluation:

- Precision: **30.6%**
- Recall: **91.7%**

The model is therefore good at recovering many true digital-content examples, but it over-predicts the category for unrelated messages.

### 4. Incorrect Resolution Transfer

Retrieval can find historically similar messages that solved a related problem but required a different action.

This can cause the generator to transfer the wrong historical resolution.

### 5. Missed Escalations and Unusual Phrasing

Some messages contain signals that the customer has already tried to resolve the issue multiple times.

Other messages contain unusual wording, malformed text, or non-English content that can produce unreliable intent predictions even when retrieval finds useful historical examples.

Detailed examples, hypotheses, and proposed mitigations are documented in:

```text
reports/report.md
```

---

## 14. What Is Misleading About My Headline Number?

The **55.0% intent accuracy** is useful, but it should not be interpreted as "the agent is correct only 55.0% of the time" across the entire product.

There are several reasons:

- The benchmark contains difficult and ambiguous Twitter messages.
- Intent classification is only one component of the system.
- The escalation policy is evaluated separately and achieves 87.5% accuracy.
- Response quality is evaluated independently using grounding, relevance, safety, style, and resolution correctness.
- Some classification mistakes do not necessarily produce unusable responses.
- The 200-example benchmark is relatively small.
- The confidence analysis shows substantial variation in reliability: predictions with confidence >= 0.85 achieved 77.2% accuracy, while low-confidence predictions were much less reliable.

Conversely, the **87.5% escalation accuracy** should also not be interpreted as proof that escalation is production-ready. Missing a single high-risk escalation can be substantially more costly than a routine classification error.

The headline numbers therefore need to be read together with the failure analysis and judge/human calibration results.

---

## 15. Key Findings

### Intent Classification

The hybrid semantic + lexical classifier improves over the simple TF-IDF baseline, but intent classification remains the weakest major component.

The hardest cases are usually short, ambiguous, or overlapping support requests.

### Retrieval

Retrieval finds relevant historical examples reasonably often, especially within the top five results.

However, semantic similarity does not guarantee that the retrieved resolution is appropriate for the current situation.

### Response Generation

The response generator produces generally safe and stylistically appropriate replies.

The main issue is transferring the **exact historical resolution** correctly.

### Escalation

The rule-based escalation policy performs better than the intent classifier on the current benchmark.

Persistence, repeated-contact, financial-risk, and security-related signals are particularly useful for deciding when a human should take over.

---

## 16. One-Week Improvement Plan

If given another week, I would prioritize:

### 1. Add Conversation Context to Intent Classification

The source data is inherently multi-turn.

A context-aware classifier could use:

```text
Previous customer/support turns
          +
Current customer message
          ↓
Context-aware representation
          ↓
Intent prediction
```

This is likely to address cases where the final customer message is short but the preceding conversation contains the actual problem.

### 2. Improve Intent Boundaries

Review the confusion matrix and create targeted training examples for the most frequently confused intent pairs.

### 3. Improve Retrieval

Move from pure semantic similarity toward hybrid retrieval using:

```text
semantic similarity
+
intent compatibility
+
keyword/entity overlap
```

### 4. Improve Resolution Grounding

Require the generator to distinguish between:

- evidence directly supported by retrieved examples
- information inferred from the customer message
- information that should not be invented

### 5. Improve Escalation and Judge Calibration

Create a larger human-labelled escalation set focused specifically on:

- repeated complaints
- unresolved refunds
- account/security issues
- payment problems
- previous support contact

Also expand the human-rated calibration set and use clearer scoring anchors for each response-quality dimension.

---

## 17. Decision Log

Non-obvious engineering decisions are documented in:

```text
reports/decision_log.md
```

The decision log covers topics including:

- Why AmazonHelp was selected
- How the intent taxonomy was derived
- Why MiniLM was chosen
- Why Logistic Regression was used
- Why the hybrid MiniLM + TF-IDF representation was used
- Why FAISS was selected
- Why golden examples are held out
- How retrieval leakage is prevented
- Why escalation uses deterministic rules
- Why Groq/Qwen is used for generation
- Why Ollama is retained as a fallback
- Why LLM-as-judge is treated as a secondary signal
- Why retrieval metrics are labelled approximate

---

## 18. Tests

Run the deterministic test suite with:

```powershell
python -m pytest -q
```

The tests cover core components including:

- Intent classification
- Retrieval
- Escalation rules
- Response generation
- Data processing
- Evaluation metrics

---

## 19. Project Structure

```text
hiver-ai-support-agent/
│
├── data/
│   ├── golden/
│   ├── processed/
│   └── raw/
│
├── evaluation/
│   ├── baselines.py
│   ├── failure_analysis.py
│   ├── human_agreement.py
│   ├── judge.py
│   ├── metrics.py
│   └── ...
│
├── reports/
│   ├── decision_log.md
│   └── report.md
│
├── src/
│   ├── agent/
│   ├── analysis/
│   ├── data/
│   ├── escalation/
│   ├── generation/
│   ├── intent/
│   └── retrieval/
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── ...
```

---

## 20. Data Note

The project uses the **Customer Support on Twitter** dataset from Kaggle.

The complete raw dataset is not committed because of its size.

The processed artifacts and golden evaluation data required for the evaluation workflow are maintained separately from the full raw dataset.

The dataset remains subject to its original source and usage terms.

---

## 21. Key Takeaway

This project demonstrates a complete AI support-agent pipeline:

```text
Classification
      ↓
Historical Retrieval
      ↓
Grounded Generation
      ↓
Escalation
      ↓
Evaluation
```

The evaluation shows that the system is strongest in **safety, response style, and escalation routing**, while the biggest opportunities are **intent disambiguation, conversation context, and exact historical-resolution transfer**.

The goal of the project is not to present a perfect support agent, but to make its capabilities, limitations, and failure modes measurable and reproducible.