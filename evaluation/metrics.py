from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def classification_metrics(y_true, y_pred, labels=None):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": report,
    }


def binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=True,
        zero_division=0,
    )

    false_auto_handles = sum(
        true and not pred
        for true, pred in zip(y_true, y_pred)
    )
    false_escalations = sum(
        not true and pred
        for true, pred in zip(y_true, y_pred)
    )

    total = len(y_true)

    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            4,
        ),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_auto_handle_rate": round(
            false_auto_handles / total,
            4,
        ),
        "false_escalation_rate": round(
            false_escalations / total,
            4,
        ),
    }
