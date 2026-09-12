import pandas as pd

from src.analysis.accept_ai_annotations import (
    BULK_NOTE,
    accept_ai_annotations,
)


def test_accept_ai_annotations_fills_unlabelled_rows():
    df = pd.DataFrame(
        [
            {
                "ai_intent": "delivery_problem",
                "ai_escalation": "false",
                "gold_intent": "",
                "expected_escalation": "",
                "annotation_notes": "",
            },
            {
                "ai_intent": "payment_billing",
                "ai_escalation": "true",
                "gold_intent": "payment_billing",
                "expected_escalation": "true",
                "annotation_notes": "Already reviewed.",
            },
        ]
    )

    accepted, skipped = accept_ai_annotations(df)

    assert accepted == 1
    assert skipped == 1
    assert df.at[0, "gold_intent"] == "delivery_problem"
    assert df.at[0, "expected_escalation"] == "false"
    assert df.at[0, "annotation_notes"] == BULK_NOTE
    assert df.at[1, "annotation_notes"] == "Already reviewed."
