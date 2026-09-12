# Hiver AI Support Agent Report

## Current Evaluation Snapshot

This project uses the Customer Support on Twitter dataset and focuses on
AmazonHelp support conversations. The current golden set contains 200 examples
sampled across a 13-intent taxonomy.

Important caveat: the current golden file is usable for development, but 187 of
the 200 labels were bulk accepted from AI/heuristic suggestions and have not yet
been individually human-reviewed. These numbers should be treated as preliminary
until the labels are audited.

## Intent Taxonomy

- delivery_problem
- order_management
- returns_refunds
- product_issue
- payment_billing
- amazon_pay
- account_access
- prime_membership
- digital_content
- product_information
- technical_issue
- support_followup
- other

## Baselines

The evaluation currently includes two baseline families:

1. Trivial baseline

   For intent classification, always predict the most common intent in the
   training fold. For escalation, always predict no escalation.

2. Simple baseline

   TF-IDF features with Logistic Regression, evaluated with 5-fold stratified
   cross-validation.

## Preliminary Results

Intent classification:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Trivial majority baseline | 0.145 | 0.0195 |
| TF-IDF + Logistic Regression | 0.395 | 0.3780 |

Escalation prediction:

| Model | Accuracy | F1 | False auto-handle rate |
| --- | ---: | ---: | ---: |
| Always no escalation | 0.895 | 0.0000 | 0.105 |
| TF-IDF + Logistic Regression | 0.890 | 0.0000 | 0.105 |

The escalation result is intentionally not sugar-coated: both baselines fail to
identify escalation-worthy examples. This suggests that escalation needs either
better labels, explicit risk rules, more positive examples, or a separate
policy-based classifier.

## What Is Misleading About My Headline Number?

The headline intent score is preliminary because most labels were bulk accepted
from AI/heuristic suggestions rather than individually human-reviewed. This can
inflate agreement with keyword-heavy baselines and may hide ambiguous examples.

Escalation accuracy is also misleading because only 21 of 200 examples are
currently labelled as escalation cases. A model can get high accuracy by almost
always predicting "do not escalate", while still being unsafe for real support
automation. For escalation, recall and false auto-handle rate matter more than
raw accuracy.

## Next Work

- Manually audit the bulk-accepted golden labels.
- Add an explicit escalation policy baseline.
- Build retrieval over non-golden AmazonHelp conversations.
- Evaluate retrieval with Recall@k / MRR.
- Add response generation and LLM-as-judge evaluation.
