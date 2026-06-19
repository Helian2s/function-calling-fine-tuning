from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any


SOURCE_DATASET = "Salesforce/xlam-function-calling-60k"
SCHEMA_VERSION = "1.0"
GENERATOR_BOUNDARY = 33_659


class NormalizationError(ValueError):
    """Raised when an xLAM record cannot be normalized safely."""


OPTIONAL_SUFFIX_PATTERN = re.compile(
    r",\s*optional\s*$",
    flags=re.IGNORECASE,
)

DEFAULT_SUFFIX_PATTERN = re.compile(
    r",\s*default\s+(?P<value>.+?)\s*$",
    flags=re.IGNORECASE,
)


SIMPLE_TYPE_SCHEMAS: dict[str, dict[str, Any]] = {
    "str": {"type": "string"},
    "string": {"type": "string"},
    "text": {"type": "string"},
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "float": {"type": "number"},
    "double": {"type": "number"},
    "number": {"type": "number"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
    "list": {
        "type": "array",
        "items": {},
    },
    "array": {
        "type": "array",
        "items": {},
    },
    "tuple": {
        "type": "array",
        "items": {},
    },
    "set": {
        "type": "array",
        "items": {},
        "uniqueItems": True,
    },
    "frozenset": {
        "type": "array",
        "items": {},
        "uniqueItems": True,
    },
    "dict": {"type": "object"},
    "dictionary": {"type": "object"},
    "mapping": {"type": "object"},
    "object": {"type": "object"},
    "null": {"type": "null"},
    "none": {"type": "null"},
    "nonetype": {"type": "null"},
    "any": {},
}

def extract_annotated_default(
    raw_type: Any,
) -> tuple[bool, Any]:
    """Extract a default value embedded in a source type annotation.

    Example:
        "str, optional, default 'London'"
        -> (True, "London")
    """
    if not isinstance(raw_type, str):
        return False, None

    match = DEFAULT_SUFFIX_PATTERN.search(raw_type.strip())

    if match is None:
        return False, None

    raw_value = match.group("value").strip()

    try:
        return True, ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise NormalizationError(
            f"Invalid annotated default value in type "
            f"{raw_type!r}."
        ) from exc

def decode_json_field(
    value: Any,
    *,
    field_name: str,
    row_id: Any,
) -> Any:
    """Decode a JSON string while accepting already-decoded values."""
    if isinstance(value, (list, dict)):
        return value

    if not isinstance(value, str):
        raise NormalizationError(
            f"Row {row_id}: {field_name} must be a JSON string, "
            f"list, or dictionary; received {type(value).__name__}."
        )

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise NormalizationError(
            f"Row {row_id}: {field_name} contains invalid JSON. "
            f"Preview: {value[:200]!r}"
        ) from exc


def _node_name(node: ast.AST) -> str | None:
    """Return the final identifier from Name or Attribute AST nodes."""
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return None


def _subscript_arguments(node: ast.Subscript) -> list[ast.AST]:
    """Return type parameters from List[T], Tuple[A, B], and similar types."""
    if isinstance(node.slice, ast.Tuple):
        return list(node.slice.elts)

    return [node.slice]


def _deduplicate_schemas(
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicate JSON Schema fragments."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()

    for schema in schemas:
        key = json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
        )

        if key not in seen:
            seen.add(key)
            unique.append(schema)

    return unique


def _make_union_schema(
    schemas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a compact JSON Schema union."""
    schemas = _deduplicate_schemas(schemas)

    # JSON Schema number accepts integer values, so integer is redundant.
    if {"type": "number"} in schemas:
        schemas = [
            schema
            for schema in schemas
            if schema != {"type": "integer"}
        ]

    if len(schemas) == 1:
        return schemas[0]

    return {"anyOf": schemas}


def _schema_from_type_node(node: ast.AST) -> dict[str, Any]:
    """Convert a parsed Python-style type expression to JSON Schema."""
    if isinstance(node, ast.Name):
        name = node.id.lower()

        if name not in SIMPLE_TYPE_SCHEMAS:
            raise NormalizationError(
                f"Unsupported parameter type name: {node.id!r}."
            )

        return copy.deepcopy(SIMPLE_TYPE_SCHEMAS[name])

    if isinstance(node, ast.Attribute):
        name = node.attr.lower()

        if name not in SIMPLE_TYPE_SCHEMAS:
            raise NormalizationError(
                f"Unsupported parameter type name: {node.attr!r}."
            )

        return copy.deepcopy(SIMPLE_TYPE_SCHEMAS[name])

    if isinstance(node, ast.Constant):
        if node.value is None:
            return {"type": "null"}

        raise NormalizationError(
            f"Unsupported type constant: {node.value!r}."
        )

    # Support PEP 604 forms such as int | float.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _make_union_schema(
            [
                _schema_from_type_node(node.left),
                _schema_from_type_node(node.right),
            ]
        )

    if not isinstance(node, ast.Subscript):
        raise NormalizationError(
            f"Unsupported type expression: {ast.dump(node)}."
        )

    container_name = _node_name(node.value)

    if container_name is None:
        raise NormalizationError(
            f"Unsupported generic type: {ast.dump(node.value)}."
        )

    container = container_name.lower()
    arguments = _subscript_arguments(node)

    if container in {"list", "array", "sequence"}:
        if len(arguments) != 1:
            raise NormalizationError(
                f"{container_name} expects one type argument."
            )

        return {
            "type": "array",
            "items": _schema_from_type_node(arguments[0]),
        }

    if container in {"set", "frozenset"}:
        if len(arguments) != 1:
            raise NormalizationError(
                f"{container_name} expects one type argument."
            )

        return {
            "type": "array",
            "items": _schema_from_type_node(arguments[0]),
            "uniqueItems": True,
        }

    if container == "tuple":
        if not arguments:
            return {
                "type": "array",
                "items": {},
            }

        # Tuple[int, ...]
        if (
            len(arguments) == 2
            and isinstance(arguments[1], ast.Constant)
            and arguments[1].value is Ellipsis
        ):
            return {
                "type": "array",
                "items": _schema_from_type_node(arguments[0]),
            }

        item_schemas = [
            _schema_from_type_node(argument)
            for argument in arguments
        ]

        # Homogeneous fixed-length tuple, such as Tuple[float, float].
        if all(schema == item_schemas[0] for schema in item_schemas):
            return {
                "type": "array",
                "items": item_schemas[0],
                "minItems": len(item_schemas),
                "maxItems": len(item_schemas),
            }

        # Heterogeneous fixed-length tuple.
        return {
            "type": "array",
            "prefixItems": item_schemas,
            "minItems": len(item_schemas),
            "maxItems": len(item_schemas),
        }

    if container == "union":
        return _make_union_schema(
            [
                _schema_from_type_node(argument)
                for argument in arguments
            ]
        )

    if container == "optional":
        if len(arguments) != 1:
            raise NormalizationError(
                "Optional expects exactly one type argument."
            )

        return _make_union_schema(
            [
                _schema_from_type_node(arguments[0]),
                {"type": "null"},
            ]
        )

    if container in {"dict", "mapping"}:
        if len(arguments) >= 2:
            value_schema = _schema_from_type_node(arguments[-1])
        else:
            value_schema = {}

        return {
            "type": "object",
            "additionalProperties": value_schema,
        }

    raise NormalizationError(
        f"Unsupported generic parameter type: {container_name!r}."
    )


def normalize_type_schema(
    raw_type: Any,
) -> tuple[dict[str, Any], bool]:
    """Return a JSON Schema fragment and an optional-property marker."""
    if isinstance(raw_type, list):
        schemas: list[dict[str, Any]] = []
        optional_marker = False

        for item in raw_type:
            schema, item_optional = normalize_type_schema(item)
            schemas.append(schema)
            optional_marker = optional_marker or item_optional

        return _make_union_schema(schemas), optional_marker

    if not isinstance(raw_type, str) or not raw_type.strip():
        raise NormalizationError(
            "Parameter type must be a non-empty string; "
            f"received {raw_type!r}."
        )

    source = raw_type.strip()

    # Remove a trailing source annotation such as:
    # ", default 'London'"
    source = DEFAULT_SUFFIX_PATTERN.sub("", source).strip()

    optional_marker = bool(
        OPTIONAL_SUFFIX_PATTERN.search(source)
    )

    source = OPTIONAL_SUFFIX_PATTERN.sub("", source).strip()

    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise NormalizationError(
            f"Invalid parameter type expression: {raw_type!r}."
        ) from exc

    return _schema_from_type_node(node), optional_marker


def normalize_type(raw_type: Any) -> str | dict[str, Any]:
    """Return the top-level JSON Schema type when one exists.

    Complex schemas without a single top-level type, such as unions
    represented with ``anyOf``, are returned as complete dictionaries.
    """
    schema, _ = normalize_type_schema(raw_type)

    schema_type = schema.get("type")

    if isinstance(schema_type, str):
        return schema_type

    return schema

def normalize_property_schema(
    raw_schema: Any,
    *,
    parameter_name: str,
) -> dict[str, Any]:
    """Normalize one xLAM parameter definition into JSON Schema."""
    if not isinstance(raw_schema, dict):
        raise NormalizationError(
            f"Parameter {parameter_name!r} must be an object; "
            f"received {type(raw_schema).__name__}."
        )

    raw_type = raw_schema.get("type")

    if raw_type is None:
        raise NormalizationError(
            f"Parameter {parameter_name!r} does not define a type."
        )

    result, _ = normalize_type_schema(raw_type)
    annotated_default_present, annotated_default = (
        extract_annotated_default(raw_type)
    )

    if (
        annotated_default_present
        and "default" not in raw_schema
    ):
        result["default"] = annotated_default

    description = raw_schema.get("description")

    if isinstance(description, str) and description.strip():
        result["description"] = description.strip()

    for key in (
        "enum",
        "default",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    ):
        if key in raw_schema:
            result[key] = raw_schema[key]

    if result.get("type") == "array":
        raw_items = raw_schema.get("items")

        if isinstance(raw_items, dict):
            result["items"] = normalize_property_schema(
                raw_items,
                parameter_name=f"{parameter_name}[]",
            )
        elif isinstance(raw_items, str):
            result["items"], _ = normalize_type_schema(raw_items)
        elif "items" not in result:
            result["items"] = {}

        if "uniqueItems" in raw_schema:
            result["uniqueItems"] = bool(
                raw_schema["uniqueItems"]
            )

    if result.get("type") == "object":
        raw_properties = raw_schema.get("properties")

        if isinstance(raw_properties, dict):
            properties: dict[str, Any] = {}
            required: list[str] = []

            for child_name, child_schema in raw_properties.items():
                child_name = str(child_name)

                properties[child_name] = normalize_property_schema(
                    child_schema,
                    parameter_name=(
                        f"{parameter_name}.{child_name}"
                    ),
                )

                if (
                    isinstance(child_schema, dict)
                    and child_schema.get("required") is True
                ):
                    required.append(child_name)

            result["properties"] = properties

            top_level_required = raw_schema.get("required")

            if isinstance(top_level_required, list):
                required.extend(
                    str(item)
                    for item in top_level_required
                )

            required = sorted(set(required))

            if required:
                result["required"] = required

        if "additionalProperties" in raw_schema:
            additional_properties = raw_schema[
                "additionalProperties"
            ]

            if isinstance(additional_properties, dict):
                result["additionalProperties"] = (
                    normalize_property_schema(
                        additional_properties,
                        parameter_name=(
                            f"{parameter_name}.*"
                        ),
                    )
                )
            else:
                result["additionalProperties"] = (
                    additional_properties
                )

    return result


def normalize_parameters(
    raw_parameters: Any,
) -> dict[str, Any]:
    """Convert xLAM function parameters into an object JSON Schema."""
    if raw_parameters is None:
        raw_parameters = {}

    if not isinstance(raw_parameters, dict):
        raise NormalizationError(
            "Function parameters must be a dictionary."
        )

    # The source may already contain an object JSON Schema.
    if (
        raw_parameters.get("type") == "object"
        and isinstance(raw_parameters.get("properties"), dict)
    ):
        normalized = normalize_property_schema(
            raw_parameters,
            parameter_name="<parameters>",
        )

        if normalized.get("type") != "object":
            raise NormalizationError(
                "Top-level function parameters must be an object."
            )

        return normalized

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter_name, raw_schema in raw_parameters.items():
        parameter_name = str(parameter_name)

        properties[parameter_name] = normalize_property_schema(
            raw_schema,
            parameter_name=parameter_name,
        )

        if (
            isinstance(raw_schema, dict)
            and raw_schema.get("required") is True
        ):
            required.append(parameter_name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        result["required"] = sorted(set(required))

    return result


def normalize_tool(
    raw_tool: Any,
    *,
    row_id: Any,
) -> dict[str, Any]:
    """Normalize one xLAM tool definition."""
    if not isinstance(raw_tool, dict):
        raise NormalizationError(
            f"Row {row_id}: tool must be an object."
        )

    if (
        raw_tool.get("type") == "function"
        and isinstance(raw_tool.get("function"), dict)
    ):
        raw_function = raw_tool["function"]
    else:
        raw_function = raw_tool

    name = raw_function.get("name")

    if not isinstance(name, str) or not name.strip():
        raise NormalizationError(
            f"Row {row_id}: tool has no valid name."
        )

    description = raw_function.get("description", "")

    if description is None:
        description = ""

    if not isinstance(description, str):
        description = str(description)

    return {
        "type": "function",
        "function": {
            "name": name.strip(),
            "description": description.strip(),
            "parameters": normalize_parameters(
                raw_function.get("parameters", {})
            ),
        },
    }


def normalize_tool_call(
    raw_answer: Any,
    *,
    row_id: Any,
    call_index: int,
    available_tool_names: set[str],
) -> dict[str, Any]:
    """Normalize one xLAM answer into an assistant tool call."""
    if not isinstance(raw_answer, dict):
        raise NormalizationError(
            f"Row {row_id}: answer {call_index} must be an object."
        )

    name = raw_answer.get("name")

    if not isinstance(name, str) or not name.strip():
        raise NormalizationError(
            f"Row {row_id}: answer {call_index} "
            "has no valid tool name."
        )

    name = name.strip()

    if name not in available_tool_names:
        raise NormalizationError(
            f"Row {row_id}: answer {call_index} references "
            f"unavailable tool {name!r}."
        )

    arguments = raw_answer.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise NormalizationError(
                f"Row {row_id}: arguments for call {call_index} "
                "contain invalid JSON."
            ) from exc

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        raise NormalizationError(
            f"Row {row_id}: arguments for call {call_index} "
            f"must be an object; received "
            f"{type(arguments).__name__}."
        )

    return {
        "id": f"call_{call_index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def normalize_xlam_row(
    row: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Normalize one raw xLAM row into canonical chat format."""
    required_fields = {"id", "query", "tools", "answers"}
    missing_fields = required_fields - set(row)

    if missing_fields:
        raise NormalizationError(
            f"Missing required fields: {sorted(missing_fields)}."
        )

    row_id = int(row["id"])
    query = row["query"]

    if not isinstance(query, str) or not query.strip():
        raise NormalizationError(
            f"Row {row_id}: query must be a non-empty string."
        )

    raw_tools = decode_json_field(
        row["tools"],
        field_name="tools",
        row_id=row_id,
    )

    raw_answers = decode_json_field(
        row["answers"],
        field_name="answers",
        row_id=row_id,
    )

    if not isinstance(raw_tools, list) or not raw_tools:
        raise NormalizationError(
            f"Row {row_id}: tools must be a non-empty list."
        )

    if not isinstance(raw_answers, list) or not raw_answers:
        raise NormalizationError(
            f"Row {row_id}: answers must be a non-empty list."
        )

    tools = [
        normalize_tool(
            raw_tool,
            row_id=row_id,
        )
        for raw_tool in raw_tools
    ]

    available_tool_names = {
        tool["function"]["name"]
        for tool in tools
    }

    if len(available_tool_names) != len(tools):
        raise NormalizationError(
            f"Row {row_id}: duplicate tool names are present."
        )

    tool_calls = [
        normalize_tool_call(
            raw_answer,
            row_id=row_id,
            call_index=index,
            available_tool_names=available_tool_names,
        )
        for index, raw_answer in enumerate(
            raw_answers,
            start=1,
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"xlam-{row_id}",
        "tools": tools,
        "messages": [
            {
                "role": "user",
                "content": query.strip(),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
        ],
        "metadata": {
            "source_dataset": SOURCE_DATASET,
            "source_id": row_id,
            "split": split,
            "generator": (
                "deepseek"
                if row_id < GENERATOR_BOUNDARY
                else "mixtral"
            ),
            "available_tool_count": len(tools),
            "expected_call_count": len(tool_calls),
        },
    }