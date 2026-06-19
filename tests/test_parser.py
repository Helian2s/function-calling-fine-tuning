from function_calling_ft.parser import parse_tool_calls


def test_parse_single_tool_call_from_json_object() -> None:
    result = parse_tool_calls(
        '{"name":"weather","arguments":{"city":"Salt Lake City"}}'
    )

    assert result.valid_structure is True
    assert len(result.calls) == 1
    assert result.calls[0].name == "weather"
    assert result.calls[0].arguments == {
        "city": "Salt Lake City"
    }


def test_parse_list_of_tool_calls() -> None:
    result = parse_tool_calls(
        """
        [
          {"name":"weather","arguments":{"city":"Salt Lake City"}},
          {"name":"calendar","arguments":{"date":"2026-06-21"}}
        ]
        """
    )

    assert result.valid_structure is True
    assert [call.name for call in result.calls] == [
        "weather",
        "calendar",
    ]


def test_parse_parallel_tool_calls_from_wrapper_object() -> None:
    result = parse_tool_calls(
        """
        {
          "tool_calls": [
            {
              "type": "function",
              "function": {
                "name": "weather",
                "arguments": {"city": "Salt Lake City"}
              }
            },
            {
              "type": "function",
              "function": {
                "name": "news",
                "arguments": {"topic": "ski"}
              }
            }
          ]
        }
        """
    )

    assert result.valid_structure is True
    assert [call.name for call in result.calls] == [
        "weather",
        "news",
    ]


def test_parse_tool_calls_tolerates_extra_prose() -> None:
    result = parse_tool_calls(
        """
        I will call the function now.

        {"arguments":{"unit":"celsius","city":"Salt Lake City"},"name":"weather"}

        Done.
        """
    )

    assert result.valid_structure is True
    assert result.had_extra_prose is True
    assert result.calls[0].arguments == {
        "city": "Salt Lake City",
        "unit": "celsius",
    }


def test_parse_tool_calls_reports_malformed_output_without_crashing() -> None:
    result = parse_tool_calls("weather(city='Salt Lake City'")

    assert result.valid_structure is False
    assert result.calls == ()
    assert "No JSON object or array found" in result.errors[0]


def test_parse_tool_calls_reports_missing_arguments() -> None:
    result = parse_tool_calls('{"name":"weather"}')

    assert result.valid_structure is False
    assert len(result.calls) == 1
    assert result.calls[0].name == "weather"
    assert result.calls[0].arguments is None
    assert "arguments is missing" in result.errors[0]


def test_parse_tool_calls_reports_non_object_arguments() -> None:
    result = parse_tool_calls(
        '{"name":"weather","arguments":["Salt Lake City"]}'
    )

    assert result.valid_structure is False
    assert len(result.calls) == 1
    assert result.calls[0].name == "weather"
    assert result.calls[0].arguments is None
    assert "arguments must be a JSON object" in result.errors[0]
