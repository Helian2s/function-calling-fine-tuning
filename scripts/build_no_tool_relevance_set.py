#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import write_jsonl


SCHEMA_VERSION = "1.0"
OUTPUT_DIR = Path("data/eval/no_tool_relevance_v1")
DATASET_NAME = "human-authored-no-tool-relevance-v1"
CATEGORIES = (
    "no_available_tool_can_satisfy",
    "available_tools_irrelevant",
    "missing_required_information",
    "direct_answer_without_tool",
)


TOOLS = {
    "weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a named city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    "stock": {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the latest stock price for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    "calendar": {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event with title, date, and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["title", "date", "time"],
            },
        },
    },
    "recipe": {
        "type": "function",
        "function": {
            "name": "find_recipe",
            "description": "Find recipes for named ingredients.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["ingredients"],
            },
        },
    },
}


DIRECT_FACTS = [
    ("What is 2 + 2?", "4"),
    ("What is 9 minus 4?", "5"),
    ("What is 6 times 7?", "42"),
    ("What is half of 18?", "9"),
    ("What is 15 divided by 3?", "5"),
    ("What is the plural of cactus?", "cacti"),
    ("What color do you get by mixing blue and yellow paint?", "green"),
    ("How many days are in a standard non-leap year?", "365"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is the first letter of the English alphabet?", "A"),
    ("Convert 100 centimeters to meters.", "1 meter"),
    ("Convert 1 kilogram to grams.", "1000 grams"),
    ("What is the opposite of north?", "south"),
    ("What is 12 squared?", "144"),
    ("What is the lowercase form of HELLO?", "hello"),
    ("How many sides does a triangle have?", "3"),
    ("How many hours are in one day?", "24"),
    ("What is 8 plus 11?", "19"),
    ("What is the next number after 99?", "100"),
    ("What is 30 percent of 10?", "3"),
    ("What punctuation ends a question?", "question mark"),
    ("What is the past tense of walk?", "walked"),
    ("How many vowels are in the word apple?", "2"),
    ("What is 5 cubed?", "125"),
    ("What is the Roman numeral for five?", "V"),
]


NO_TOOL_REQUESTS = [
    ("Translate 'good morning' into Pig Latin.", "oodgay orningmay"),
    ("Write a three-word title for a poem about rain.", "Rain Over Windows"),
    ("Alphabetize these words: pear, apple, banana.", "apple, banana, pear"),
    ("Rewrite 'the cat sleeps' in uppercase.", "THE CAT SLEEPS"),
    ("Give one synonym for quick.", "fast"),
    ("Count the letters in the word orbit.", "5"),
    ("Make the sentence 'birds fly' past tense.", "birds flew"),
    ("Name one primary color.", "red"),
    ("What is the third month of the year?", "March"),
    ("Reverse the word level.", "level"),
    ("What is the abbreviation for tablespoon?", "tbsp"),
    ("Give an antonym for empty.", "full"),
    ("What is 4 factorial?", "24"),
    ("Which is larger, 0.8 or 0.75?", "0.8"),
    ("Round 3.6 to the nearest whole number.", "4"),
    ("What is the square root of 81?", "9"),
    ("Spell the word queue.", "queue"),
    ("Make 'analysis' plural.", "analyses"),
    ("What is the last letter of zebra?", "a"),
    ("How many minutes are in two hours?", "120"),
    ("Change 'I am' to a contraction.", "I'm"),
    ("What is one half written as a decimal?", "0.5"),
    ("What is the sum of angles in a triangle?", "180 degrees"),
    ("Remove spaces from 'fine tuning'.", "finetuning"),
    ("What is the opposite of ancient?", "modern"),
]


IRRELEVANT_TOOL_REQUESTS = [
    ("Should I use a comma before a coordinating conjunction?", "Use a comma before a coordinating conjunction when it joins two independent clauses."),
    ("What is a polite way to decline an invitation?", "Thank you for inviting me, but I cannot attend."),
    ("Give a short definition of evaporation.", "Evaporation is the process where liquid changes into vapor."),
    ("What does CPU stand for?", "central processing unit"),
    ("Name one benefit of writing tests.", "Tests help catch regressions."),
    ("What is a haiku?", "A haiku is a short poem traditionally written in three lines."),
    ("Explain what a variable is in programming.", "A variable is a named place to store a value."),
    ("What is a simple greeting for an email?", "Hello,"),
    ("Give one reason to back up files.", "Backups help recover data after loss or damage."),
    ("What does HTML stand for?", "HyperText Markup Language"),
    ("What is a noun?", "A noun names a person, place, thing, or idea."),
    ("Give one example of a mammal.", "human"),
    ("What is the freezing point of water in Celsius?", "0 degrees Celsius"),
    ("What is the boiling point of water in Celsius?", "100 degrees Celsius"),
    ("Explain what version control is.", "Version control tracks changes to files over time."),
    ("What is a checksum used for?", "A checksum helps verify data integrity."),
    ("Define photosynthesis simply.", "Photosynthesis is how plants use light to make food."),
    ("What is a byte?", "A byte is commonly eight bits."),
    ("What is a keyboard shortcut for copy on Windows?", "Ctrl+C"),
    ("What is a keyboard shortcut for paste on Windows?", "Ctrl+V"),
    ("Give one example of renewable energy.", "solar power"),
    ("What is a paragraph?", "A paragraph is a group of related sentences."),
    ("What does API stand for?", "application programming interface"),
    ("What is a decimal number?", "A decimal number uses base ten place values."),
    ("What is a prime number?", "A prime number has exactly two positive divisors."),
]


MISSING_INFO_REQUESTS = [
    ("Book a meeting for tomorrow.", "What title and time should I use for the meeting?"),
    ("Check the weather for me.", "Which city should I check the weather for?"),
    ("Find a recipe for dinner.", "Which ingredients should I use for the recipe search?"),
    ("Get the stock price.", "Which ticker symbol should I look up?"),
    ("Create a calendar event.", "What title, date, and time should I use for the event?"),
    ("Tell me the forecast.", "Which city should I use for the forecast?"),
    ("Look up a share price.", "Which ticker symbol should I look up?"),
    ("Schedule lunch.", "What date and time should I schedule lunch for?"),
    ("Find something I can cook.", "Which ingredients do you want to use?"),
    ("Add an appointment.", "What title, date, and time should I use for the appointment?"),
    ("How is the weather there?", "Which city should I check the weather for?"),
    ("Get market data.", "Which ticker symbol should I look up?"),
    ("Plan a call.", "What title, date, and time should I use for the call?"),
    ("Search for a recipe.", "Which ingredients should I use for the recipe search?"),
    ("What is the temperature?", "Which city should I check the temperature for?"),
    ("Set up a reminder.", "What title, date, and time should I use for the reminder?"),
    ("Find a meal idea.", "Which ingredients do you want to use?"),
    ("Check a ticker.", "Which ticker symbol should I look up?"),
    ("Put something on my calendar.", "What title, date, and time should I use?"),
    ("Is it raining?", "Which city should I check for rain?"),
    ("Find stock information.", "Which ticker symbol should I look up?"),
    ("Schedule a project review.", "What date and time should I schedule the project review for?"),
    ("Suggest a recipe.", "Which ingredients should I use for the recipe search?"),
    ("Check conditions outside.", "Which city should I check weather conditions for?"),
    ("Add a calendar entry.", "What title, date, and time should I use for the calendar entry?"),
]


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record(
    *,
    split: str,
    category: str,
    index: int,
    user_query: str,
    expected: str,
    tools: list[dict[str, Any]],
    expected_type: str,
) -> dict[str, Any]:
    record_id = f"no-tool-rel-v1-{split}-{category}-{index:03d}"
    split_lock_status = (
        "locked_final_no_tool" if split == "final" else "screening_allowed"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "messages": [{"role": "user", "content": user_query}],
        "tools": tools,
        "expected_response": {
            "type": expected_type,
            "content": expected,
        },
        "metadata": {
            "source_dataset": DATASET_NAME,
            "source_id": f"{split}-{category}-{index:03d}",
            "split": f"no_tool_{split}",
            "available_tool_count": len(tools),
            "expected_call_count": 0,
            "normalization_warnings": [],
        },
        "relevance_metadata": {
            "category": category,
            "source": "human_authored",
            "review_status": "reviewable",
            "query_hash": _stable_hash(user_query),
            "tool_names": [
                tool["function"]["name"]
                for tool in tools
                if isinstance(tool.get("function"), dict)
            ],
        },
        "split_metadata": {
            "primary_split": f"no_tool_{split}",
            "split_lock_status": split_lock_status,
            "split_schema_version": SCHEMA_VERSION,
            "subset_name": f"no_tool_{split}",
        },
    }


def build_records(split: str) -> list[dict[str, Any]]:
    offset = 0 if split == "dev" else 1000
    records: list[dict[str, Any]] = []
    for index, (query, expected) in enumerate(NO_TOOL_REQUESTS, start=1):
        records.append(
            _record(
                split=split,
                category="no_available_tool_can_satisfy",
                index=index + offset,
                user_query=query,
                expected=expected,
                tools=[TOOLS["weather"], TOOLS["stock"]],
                expected_type="direct_answer",
            )
        )
    for index, (query, expected) in enumerate(IRRELEVANT_TOOL_REQUESTS, start=1):
        records.append(
            _record(
                split=split,
                category="available_tools_irrelevant",
                index=index + offset,
                user_query=query,
                expected=expected,
                tools=[TOOLS["weather"], TOOLS["calendar"], TOOLS["recipe"]],
                expected_type="direct_answer",
            )
        )
    for index, (query, expected) in enumerate(MISSING_INFO_REQUESTS, start=1):
        records.append(
            _record(
                split=split,
                category="missing_required_information",
                index=index + offset,
                user_query=query,
                expected=expected,
                tools=[TOOLS["weather"], TOOLS["stock"], TOOLS["calendar"], TOOLS["recipe"]],
                expected_type="clarification",
            )
        )
    for index, (query, expected) in enumerate(DIRECT_FACTS, start=1):
        records.append(
            _record(
                split=split,
                category="direct_answer_without_tool",
                index=index + offset,
                user_query=query,
                expected=expected,
                tools=[TOOLS["weather"], TOOLS["stock"]],
                expected_type="direct_answer",
            )
        )
    return sorted(records, key=lambda record: str(record["id"]))


def _write_checksums(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*.json*")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.name}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = build_records("dev")
    final = build_records("final")
    if len(dev) != 100 or len(final) != 100:
        raise SystemExit("expected 100 dev and 100 final no-tool records")
    write_jsonl(output_dir / "dev.jsonl", dev)
    write_jsonl(output_dir / "final_locked.jsonl", final)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "dev_records": len(dev),
        "final_records": len(final),
        "categories": {
            category: {
                "dev": sum(
                    int(record["relevance_metadata"]["category"] == category)
                    for record in dev
                ),
                "final": sum(
                    int(record["relevance_metadata"]["category"] == category)
                    for record in final
                ),
            }
            for category in CATEGORIES
        },
        "final_lock_status": "locked_final_no_tool",
        "construction": "deterministic human-authored templates",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output_dir)
    print(f"dev={output_dir / 'dev.jsonl'}")
    print(f"final_locked={output_dir / 'final_locked.jsonl'}")
    print(f"manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
