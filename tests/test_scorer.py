from function_calling_ft.scorer import score_call, score_calls


def test_argument_order_does_not_matter() -> None:
    expected = {
        "name": "weather",
        "arguments": {
            "city": "Salt Lake City",
            "unit": "celsius",
        },
    }

    predicted = {
        "name": "weather",
        "arguments": {
            "unit": "celsius",
            "city": "Salt Lake City",
        },
    }

    assert score_call(predicted, expected).complete_match


def test_score_call_rejects_unknown_function_name() -> None:
    expected = {
        "name": "weather",
        "arguments": {"city": "Salt Lake City"},
    }
    predicted = {
        "name": "forecast",
        "arguments": {"city": "Salt Lake City"},
    }

    score = score_call(predicted, expected)

    assert score.valid_structure is True
    assert score.correct_function_name is False
    assert score.complete_match is False


def test_score_call_rejects_missing_arguments() -> None:
    expected = {
        "name": "weather",
        "arguments": {"city": "Salt Lake City"},
    }
    predicted = {"name": "weather"}

    score = score_call(predicted, expected)

    assert score.valid_structure is False
    assert score.correct_function_name is True
    assert score.correct_argument_names is False
    assert score.correct_argument_values is False
    assert score.complete_match is False


def test_score_call_handles_malformed_output_without_crashing() -> None:
    expected = {
        "name": "weather",
        "arguments": {"city": "Salt Lake City"},
    }

    score = score_call(
        "I think the answer is weather(city='Salt Lake City')",
        expected,
    )

    assert score.valid_structure is False
    assert score.complete_match is False
    assert score.parse_errors


def test_score_calls_ignores_parallel_call_order_by_default() -> None:
    expected = [
        {
            "name": "weather",
            "arguments": {"city": "Salt Lake City"},
        },
        {
            "name": "news",
            "arguments": {"topic": "ski"},
        },
    ]
    predicted = [
        {
            "name": "news",
            "arguments": {"topic": "ski"},
        },
        {
            "name": "weather",
            "arguments": {"city": "Salt Lake City"},
        },
    ]

    score = score_calls(predicted, expected)

    assert score.valid_structure is True
    assert score.complete_match is True


def test_score_calls_can_require_order_when_requested() -> None:
    expected = [
        {
            "name": "weather",
            "arguments": {"city": "Salt Lake City"},
        },
        {
            "name": "news",
            "arguments": {"topic": "ski"},
        },
    ]
    predicted = [
        {
            "name": "news",
            "arguments": {"topic": "ski"},
        },
        {
            "name": "weather",
            "arguments": {"city": "Salt Lake City"},
        },
    ]

    score = score_calls(
        predicted,
        expected,
        order_matters=True,
    )

    assert score.valid_structure is True
    assert score.complete_match is False
    assert score.correct_function_name is False


def test_score_calls_handles_extra_prose_and_whitespace() -> None:
    predicted = """
    I will make both calls.

    [
      {
        "arguments": {"city": "Salt Lake City"},
        "name": "weather"
      },
      {
        "name": "news",
        "arguments": {"topic": "ski"}
      }
    ]
    """
    expected = [
        {
            "name": "weather",
            "arguments": {"city": "Salt Lake City"},
        },
        {
            "name": "news",
            "arguments": {"topic": "ski"},
        },
    ]

    score = score_calls(predicted, expected)

    assert score.valid_structure is True
    assert score.complete_match is True
