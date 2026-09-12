from src.analysis.llm_annotator import (
    AnnotationError,
    extract_json_object,
    validate_annotation,
)


VALID_CANDIDATE = "delivery_problem"


def test_extract_json_object_handles_extra_text():
    data = extract_json_object(
        'Here is the label: {"intent": "delivery_problem", '
        '"escalation": false, "confidence": 91, '
        '"reason": "The package has not arrived."}'
    )

    assert data["intent"] == "delivery_problem"
    assert data["confidence"] == 91


def test_validate_annotation_accepts_valid_response():
    suggestion = validate_annotation(
        {
            "intent": "delivery_problem",
            "escalation": False,
            "confidence": 91,
            "reason": "The package has not arrived.",
        },
        candidate_intent=VALID_CANDIDATE,
        source="test",
    )

    assert suggestion.intent == "delivery_problem"
    assert suggestion.escalation is False
    assert suggestion.confidence == 91
    assert suggestion.reason == "The package has not arrived."
    assert suggestion.disagrees_with_candidate is False
    assert suggestion.low_confidence is False


def test_validate_annotation_flags_candidate_disagreement():
    suggestion = validate_annotation(
        {
            "intent": "order_management",
            "escalation": False,
            "confidence": 65,
            "reason": "The customer wants to cancel an order.",
        },
        candidate_intent="delivery_problem",
        source="test",
    )

    assert suggestion.intent == "order_management"
    assert suggestion.disagrees_with_candidate is True
    assert suggestion.low_confidence is True


def test_validate_annotation_rejects_invalid_intent():
    try:
        validate_annotation(
            {
                "intent": "not_a_real_intent",
                "escalation": False,
                "confidence": 90,
                "reason": "Invalid intent test.",
            },
            candidate_intent=VALID_CANDIDATE,
            source="test",
        )
    except AnnotationError as error:
        assert "Invalid intent" in str(error)
    else:
        raise AssertionError("Expected AnnotationError")


def test_validate_annotation_rejects_missing_keys():
    try:
        validate_annotation(
            {
                "intent": "delivery_problem",
                "escalation": False,
                "confidence": 90,
            },
            candidate_intent=VALID_CANDIDATE,
            source="test",
        )
    except AnnotationError as error:
        assert "Missing keys" in str(error)
    else:
        raise AssertionError("Expected AnnotationError")


def test_validate_annotation_rejects_invalid_confidence():
    try:
        validate_annotation(
            {
                "intent": "delivery_problem",
                "escalation": False,
                "confidence": 101,
                "reason": "Confidence is out of range.",
            },
            candidate_intent=VALID_CANDIDATE,
            source="test",
        )
    except AnnotationError as error:
        assert "between 0 and 100" in str(error)
    else:
        raise AssertionError("Expected AnnotationError")