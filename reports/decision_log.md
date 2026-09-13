# Decision Log

This log records non-obvious engineering decisions made during development and the reasoning behind them.

## 1. Use AmazonHelp as the target brand

**Decision:** Build the initial system for AmazonHelp rather than supporting all brands in the dataset.

**Why:** AmazonHelp provides enough volume and issue diversity to build and evaluate a meaningful support agent while keeping the first implementation tractable. The architecture remains brand-extensible.

---

## 2. Reconstruct conversations using tweet relationships

**Decision:** Use `response_tweet_id` and `in_response_to_tweet_id` rather than relying only on tweet ordering.

**Why:** The dataset is not guaranteed to be chronologically ordered in a way that preserves complete conversations. Relationship fields provide a more reliable way to connect customer messages with subsequent brand responses.

---

## 3. Define a small intent taxonomy instead of using raw tweet topics

**Decision:** Use 13 operational intents:
`delivery_problem`, `order_management`, `returns_refunds`, `product_issue`, `payment_billing`, `amazon_pay`, `account_access`, `prime_membership`, `digital_content`, `product_information`, `technical_issue`, `support_followup`, and `other`.

**Why:** The assignment asks for a small set of intents. The taxonomy was derived from recurring support patterns in the selected brand data rather than creating a class for every topic.

---

## 4. Add an explicit `other` class

**Decision:** Include `other` as a trained intent rather than forcing unmatched messages into an operational category.

**Why:** Some messages are feedback, praise, feature requests, unsupported-language text, or otherwise do not correspond cleanly to an operational intent. Forcing these into another class creates misleading predictions.

---

## 5. Exclude golden examples from training and retrieval

**Decision:** Remove golden-set examples from both classifier training data and the historical retrieval corpus.

**Why:** This prevents evaluation leakage. Otherwise, retrieval could find the exact evaluation conversation and produce artificially strong results.

---

## 6. Use MiniLM embeddings for semantic intent classification

**Decision:** Use `all-MiniLM-L6-v2` embeddings followed by logistic regression.

**Why:** A semantic classifier should handle paraphrases better than purely lexical matching. MiniLM provides a lightweight local embedding model that can run without a paid API.

---

## 7. Keep TF-IDF logistic regression as a baseline

**Decision:** Retain a simple TF-IDF + logistic regression classifier as the primary non-trivial baseline.

**Why:** It provides a useful comparison against the semantic model and demonstrates whether the additional embedding-based approach provides measurable benefit.

---

## 8. Use FAISS for historical-resolution retrieval

**Decision:** Use FAISS with normalized MiniLM embeddings and inner-product similarity.

**Why:** The system needs to retrieve historically similar customer issues and resolutions. FAISS provides fast local vector search without requiring a hosted vector database.

---

## 9. Ground generated replies in retrieved historical resolutions

**Decision:** Generate responses using retrieved historical customer-resolution pairs rather than asking the LLM to answer from general knowledge alone.

**Why:** The assignment specifically asks for replies grounded in how the brand historically resolved similar issues. Historical examples therefore serve as the primary evidence source.

---

## 10. Use a local LLM for response generation

**Decision:** Use Ollama with `llama3.2:3b`.

**Why:** This keeps the complete pipeline locally runnable and avoids paid API dependencies. It also makes reproduction easier for an evaluator.

---

## 11. Separate escalation from intent classification

**Decision:** Implement escalation as a separate policy rather than encoding escalation directly into the intent classifier.

**Why:** Intent answers "what kind of issue is this?", while escalation answers "should automation handle it?". A customer can have the same intent but require different handling depending on severity, persistence, previous support attempts, or security signals.

---

## 12. Use conservative escalation signals

**Decision:** Escalate based on explicit risk, repeated unsuccessful contact, persistent unresolved problems, and insufficient evidence rather than simply escalating particular intents.

**Why:** A blanket rule such as "all account issues require escalation" produced unnecessary escalations. The final policy instead looks for cross-intent signals indicating that automated handling is unsafe or unlikely to resolve the issue.

---

## 13. Add safety overrides for security-sensitive language

**Decision:** Give explicit security signals higher priority than normal classifier confidence.

**Why:** Messages involving phishing, unauthorized access, compromised accounts, fraud, or similar risks should not be auto-handled merely because the classifier assigns them a different intent.

---

## 14. Evaluate response quality separately from classification quality

**Decision:** Use an LLM-as-a-judge evaluation for generated responses in addition to intent and escalation metrics.

**Why:** A system can classify an issue correctly while still producing a poor or non-resolving response. Response evaluation therefore measures grounding, resolution correctness, relevance, safety, style, and overall quality separately.

---

## 15. Report weaknesses instead of optimizing only the headline score

**Decision:** Report failure modes and limitations alongside aggregate metrics.

**Why:** The assignment explicitly emphasizes proof over the system itself. Aggregate accuracy can hide weaknesses such as ambiguous intent boundaries, unsupported-language messages, low-confidence predictions, and responses that are safe but do not fully resolve the customer's problem.
