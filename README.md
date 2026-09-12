# Hiver AI Support Agent

AI customer-support agent prototype for the Hiver SDE Intern assignment, using
AmazonHelp conversations from the Customer Support on Twitter dataset.

## Current Status

- AmazonHelp selected as the target brand.
- 200-example golden set created at `data/golden/golden_dataset.csv`.
- AI-assisted annotation tooling is available.
- Baseline evaluation is implemented.

Important: the current golden labels are complete, but most were bulk accepted
from AI/heuristic suggestions. They are suitable for development, but should be
manually audited before presenting final results as hand-labelled.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas scikit-learn pytest
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Annotate Golden Set

Generate AI suggestions:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.annotate_golden --prelabel-only
```

Review labels:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.annotate_golden
```

Bulk accept AI suggestions for development only:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.accept_ai_annotations
```

## Run Baseline Evaluation

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate
```

Current preliminary results:

| Task | Baseline | Accuracy | Macro/F1 |
| --- | --- | ---: | ---: |
| Intent | Majority intent | 0.145 | 0.0195 macro F1 |
| Intent | TF-IDF + Logistic Regression | 0.395 | 0.3780 macro F1 |
| Escalation | Always no escalation | 0.895 | 0.0000 F1 |
| Escalation | TF-IDF + Logistic Regression | 0.890 | 0.0000 F1 |

Detailed results are saved to `reports/baseline_results.json`.
