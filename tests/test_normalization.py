import json

import pytest

from function_calling_ft.normalization import (
    NormalizationError,
    normalize_type,
    normalize_xlam_row,
    normalize_type_schema,
)


def make_raw_row() -> dict:
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "city": {
                    "type": "str",
                    "description": "City name",
                    "required": True,
                },
                "days": {
                    "type": "int",
                    "description": "Forecast length",
                    "required": False,
                },
            },
        }
    ]

    answers = [
        {
            "name": "get_weather",
            "arguments": {
                "city": "Boston",
                "days": 3,
            },
        }
    ]

    return {
        "id": 123,
        "query": "Give me a three-day forecast for Boston.",
        "tools": json.dumps(tools),
        "answers": json.dumps(answers),
    }


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("str", "string"),
        ("string", "string"),
        ("int", "integer"),
        ("float", "number"),
        ("bool", "boolean"),
        ("list", "array"),
        ("dict", "object"),
    ],
)
def test_normalize_type(raw_type: str, expected: str) -> None:
    assert normalize_type(raw_type) == expected


def test_normalize_single_call_row() -> None:
    result = normalize_xlam_row(
        make_raw_row(),
        split="train",
    )

    assert result["id"] == "xlam-123"
    assert result["schema_version"] == "1.0"

    function = result["tools"][0]["function"]

    assert function["name"] == "get_weather"
    assert (
        function["parameters"]["properties"]["city"]["type"]
        == "string"
    )
    assert (
        function["parameters"]["properties"]["days"]["type"]
        == "integer"
    )
    assert function["parameters"]["required"] == ["city"]

    assistant = result["messages"][1]
    call = assistant["tool_calls"][0]

    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"]["city"] == "Boston"
    assert isinstance(call["function"]["arguments"], dict)


def test_rejects_unknown_answer_tool() -> None:
    row = make_raw_row()

    row["answers"] = json.dumps(
        [
            {
                "name": "unknown_tool",
                "arguments": {},
            }
        ]
    )

    with pytest.raises(
        NormalizationError,
        match="unavailable tool",
    ):
        normalize_xlam_row(row, split="train")


def test_query_remains_plain_text() -> None:
    row = make_raw_row()

    result = normalize_xlam_row(row, split="test")

    assert (
        result["messages"][0]["content"]
        == "Give me a three-day forecast for Boston."
    )


def test_multiple_answers_become_multiple_tool_calls() -> None:
    row = make_raw_row()

    row["answers"] = json.dumps(
        [
            {
                "name": "get_weather",
                "arguments": {"city": "Boston"},
            },
            {
                "name": "get_weather",
                "arguments": {"city": "Denver"},
            },
        ]
    )

    result = normalize_xlam_row(row, split="train")
    calls = result["messages"][1]["tool_calls"]

    assert len(calls) == 2
    assert calls[0]["id"] == "call_1"
    assert calls[1]["id"] == "call_2"

@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "str, optional",
            {"type": "string"},
        ),
        (
            "int, optional",
            {"type": "integer"},
        ),
        (
            "float, optional",
            {"type": "number"},
        ),
        (
            "bool, optional",
            {"type": "boolean"},
        ),
        (
            "List[Union[int, float]]",
            {
                "type": "array",
                "items": {"type": "number"},
            },
        ),
        (
            "Tuple[float, float]",
            {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        ),
        (
            "List[List[str]]",
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        ),
        (
            "List[List[int]]",
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        ),
        (
            "set",
            {
                "type": "array",
                "items": {},
                "uniqueItems": True,
            },
        ),
        (
            "List[Tuple[int, int]]",
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        ),
        (
            "List[Tuple[float, float]]",
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        ),
    ],
)
def test_normalize_complex_type_schema(
    source: str,
    expected: dict,) -> None:
    schema, _ = normalize_type_schema(source)

    assert schema == expected

def test_optional_suffix_is_detected() -> None:
    schema, optional = normalize_type_schema(
        "str, optional"
    )

    assert schema == {"type": "string"}
    assert optional is True

def test_default_embedded_in_type_annotation() -> None:
    row = make_raw_row()

    tools = json.loads(row["tools"])
    tools[0]["parameters"]["city"] = {
        "type": "str, optional, default 'London'",
        "description": "City name",
        "required": False,
    }
    row["tools"] = json.dumps(tools)

    result = normalize_xlam_row(
        row,
        split="train",
    )

    city_schema = (
        result["tools"][0]["function"]
        ["parameters"]["properties"]["city"]
    )

    assert city_schema["type"] == "string"
    assert city_schema["default"] == "London"

    required = (
        result["tools"][0]["function"]
        ["parameters"].get("required", [])
    )

    assert "city" not in required