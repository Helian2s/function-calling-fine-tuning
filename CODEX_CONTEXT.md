# Function-Calling Fine-Tuning Project — Codex Context Pack

## Project goal

Build a reproducible hands-on fine-tuning pipeline for function calling using:

* Dataset: `Salesforce/xlam-function-calling-60k`
* Target model: Qwen3-8B
* Training framework: NVIDIA NeMo AutoModel
* Fine-tuning methods: LoRA and QLoRA
* Cloud environment: AWS EC2 with NVIDIA GPUs

The immediate work is local dataset preparation. No GPU work is required for the current task.

## Repository

Repository root:

```text
function-calling-fine-tuning/
```

Relevant structure:

```text
function-calling-fine-tuning/
├── data/
│   ├── manifests/
│   │   ├── smoke_v1_selection.json
│   │   ├── smoke_v1_summary.json
│   │   └── smoke_v1_normalization_report.json
│   └── smoke/
│       ├── raw/
│       │   ├── train.jsonl
│       │   ├── validation.jsonl
│       │   └── test.jsonl
│       └── normalized/
├── scripts/
│   ├── inspect_xlam.py
│   ├── normalize_xlam.py
│   └── select_smoke_sample.py
├── src/function_calling_ft/
│   └── normalization.py
└── tests/
    ├── test_normalization.py
    ├── test_splits.py
    └── other tests
```

## Current pipeline stages

### A2 — raw dataset inspection

Completed.

The raw dataset contains 60,000 rows with these fields:

```text
id
query
tools
answers
```

Field formats:

* `query`: ordinary natural-language Python string
* `tools`: JSON serialized inside a string
* `answers`: JSON serialized inside a string

Do not run `json.loads()` on `query`.

### A3 — deterministic stratified selection

Completed, but currently needs a robustness fix.

A3 selects exactly 200 examples:

```text
train:       120
validation:   40
test:         40
```

Primary call-count quotas:

```text
single_call:          120 total
two_calls:             50 total
three_or_more_calls:   30 total
```

The sample is also balanced between the two source-generator partitions:

```text
deepseek: 100
mixtral:  100
```

The selection must remain deterministic with seed `42`.

### A4 — normalization

The normalizer converts raw xLAM records into a canonical representation containing:

```text
schema_version
id
tools
messages
metadata
```

Tool definitions are converted to JSON Schema.

Assistant answers become structured `tool_calls`.

The normalizer currently supports:

* Scalar types: `str`, `int`, `float`, `bool`
* Optional scalar annotations such as `str, optional`
* Lists and nested lists
* Tuples
* Sets
* Unions
* Defaults embedded in type annotations
* Objects and dictionaries
* Multiple tool calls

The normalizer intentionally rejects:

* Duplicate available tool names
* Python `Callable` parameter types
* Reference calls to unavailable tools
* Invalid or unsupported source schemas

## Current issue

After regenerating the A3 sample, A4 still fails on four newly selected records:

```text
46960 — duplicate tool names
56901 — unsupported Callable parameter
48306 — duplicate tool names
9128  — unsupported Callable parameter
```

Previously, four different records with the same structural problems were manually excluded. Hard-coded source-ID exclusions are therefore not a scalable solution.

The selection logic currently allows records into the A3 candidate pool before proving that they satisfy the A4 normalization contract.

## Required architectural fix

A3 must use the actual A4 normalizer as an eligibility gate.

The desired flow is:

```text
scan raw dataset
→ parse candidate metadata
→ attempt normalize_xlam_row(...)
→ reject candidate if normalization fails
→ place only normalization-compatible candidates into selection pools
→ deterministically select 200 records
```

This ensures that every record selected by A3 can later be normalized by A4.

## Relevant APIs

Normalization code is in:

```text
src/function_calling_ft/normalization.py
```

Relevant symbols:

```python
from function_calling_ft.normalization import (
    NormalizationError,
    normalize_xlam_row,
)
```

Expected use during candidate collection:

```python
try:
    normalize_xlam_row(
        candidate["raw_row"],
        split="selection",
    )
except (NormalizationError, TypeError, ValueError) as exc:
    # Classify and count rejection.
    continue
```

The returned normalized object can be discarded during A3. This call is only an eligibility check.

The temporary split name `"selection"` is acceptable because the result is discarded. Final normalized records still receive their actual `train`, `validation`, or `test` split during A4.

## Rejection categorization

Add a helper that maps normalizer exceptions to stable rejection categories.

At minimum, support:

```text
duplicate_tool_names
unsupported_callable_parameter
unsupported_parameter_type
invalid_parameter_type_expression
unknown_answer_tool
other_normalization_error
```

Example categorization rules:

```python
message = str(error).lower()

if "duplicate tool names" in message:
    return "duplicate_tool_names"

if "callable" in message:
    return "unsupported_callable_parameter"

if "unsupported parameter type" in message:
    return "unsupported_parameter_type"

if "invalid parameter type expression" in message:
    return "invalid_parameter_type_expression"

if "unavailable tool" in message:
    return "unknown_answer_tool"

return "other_normalization_error"
```

The A3 manifest should expose counts such as:

```json
{
  "normalization_duplicate_tool_names": 12,
  "normalization_unsupported_callable_parameter": 8
}
```

The exact counts depend on the full 60,000-row dataset.

## Required changes

Primary file:

```text
scripts/select_smoke_sample.py
```

Required modifications:

1. Import `NormalizationError` and `normalize_xlam_row`.
2. Add a stable normalizer-error categorization helper.
3. During `collect_candidates()`, invoke the normalizer before adding a candidate to the valid candidate pool.
4. Reject and count candidates that cannot normalize.
5. Remove any hard-coded source-ID exclusion list.
6. Preserve all existing stratification quotas.
7. Preserve deterministic behavior.
8. Preserve duplicate-ID and fingerprint protections.
9. Preserve the 120/40/40 split sizes.
10. Preserve generator and call-count balancing.

Tests should be added or updated to verify:

* A candidate with duplicate tool names is rejected during A3 candidate collection.
* A candidate containing a `Callable` parameter is rejected.
* A normal valid candidate remains eligible.
* Rejection categories are stable.
* The selector still produces unique, disjoint splits.
* Repeated runs with the same seed produce identical selections.

## Constraints

Do not:

* Add more hard-coded source IDs.
* Change `Callable` into `string`, `object`, or unconstrained `{}`.
* Silently deduplicate tools with duplicate names.
* Modify raw xLAM records.
* Skip failed records during A4 while producing partial output.
* Change the 120/40/40 split contract.
* Change the selection seed.
* Add Qwen rendering, tokenization, or loss masking in this task.
* Modify the core normalizer unless a concrete defect is discovered.

## Expected behavior after the fix

Regenerating A3 should create:

```text
data/smoke/raw/train.jsonl        120 records
data/smoke/raw/validation.jsonl    40 records
data/smoke/raw/test.jsonl          40 records
```

Running A4 should then succeed:

```text
Normalization completed successfully.
Train: 120
Validation: 40
Test: 40
```

The normalization report should contain:

```json
{
  "total_input_records": 200,
  "total_normalized_records": 200,
  "total_errors": 0
}
```

## Commands

Run tests:

```bash
source .venv/bin/activate
pytest tests/test_normalization.py tests/test_splits.py -v
```

Regenerate A3:

```bash
rm -rf data/smoke/raw data/smoke/normalized

rm -f \
  data/manifests/smoke_v1_selection.json \
  data/manifests/smoke_v1_summary.json \
  data/manifests/smoke_v1_normalization_report.json

python scripts/select_smoke_sample.py
```

Verify counts:

```bash
wc -l \
  data/smoke/raw/train.jsonl \
  data/smoke/raw/validation.jsonl \
  data/smoke/raw/test.jsonl
```

Run A4:

```bash
python scripts/normalize_xlam.py
```

Verify the normalization report:

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path(
        "data/manifests/smoke_v1_normalization_report.json"
    ).read_text(encoding="utf-8")
)

assert report["total_input_records"] == 200
assert report["total_normalized_records"] == 200
assert report["total_errors"] == 0

assert report["splits"]["train"]["normalized_records"] == 120
assert report["splits"]["validation"]["normalized_records"] == 40
assert report["splits"]["test"]["normalized_records"] == 40

print("PASS: all 200 records normalized successfully.")
PY
```

## Definition of done

The task is complete when:

```text
[ ] No hard-coded source-ID exclusion list remains
[ ] A3 filters candidates through normalize_xlam_row()
[ ] Normalization failures are counted by stable category
[ ] A3 selects exactly 120/40/40 records
[ ] All split IDs are unique and disjoint
[ ] Selection remains deterministic
[ ] A4 normalizes all 200 records
[ ] Normalization report has zero errors
[ ] Relevant tests pass
[ ] No raw or normalized dataset content is staged for Git
```
