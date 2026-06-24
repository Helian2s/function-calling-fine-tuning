# Experiment 0 Evaluation

Evaluation is offline and deterministic. Generation writes raw model outputs
first; parsing and scoring are a separate step so malformed model text is
preserved and scored as failure instead of aborting the run.

## Prediction Records

Each JSONL prediction record must include:

```json
{
  "id": "xlam-...",
  "source_id": 123,
  "model_name": "Qwen/Qwen3-8B",
  "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
  "adapter_path": null,
  "raw_generation": "{\"name\":\"...\",\"arguments\":{...}}",
  "prompt_token_count": 1234,
  "generated_token_count": 87,
  "generation_error": null
}
```

If inference fails for one record, the generator still emits that record with an
empty `raw_generation` and a non-null `generation_error`.

## Prompt Construction

The normalized xLAM smoke record contains the user prompt and the target
assistant tool-call message. Inference removes that final target assistant
message, preserves the `tools` array, and renders the prompt with
`add_generation_prompt=True`. Qwen thinking is disabled with
`enable_thinking=False` when the tokenizer supports that argument.

## Parsing and Scoring

`function_calling_ft.parser.parse_tool_calls()` accepts common function-call
shapes including single objects, lists, wrapper keys such as `tool_calls`,
parallel calls, stringified `arguments`, and JSON with extra prose around it.
Malformed or empty output produces parse errors but does not stop evaluation.

`function_calling_ft.scorer.score_calls()` compares predicted calls with
`extract_expected_tool_calls()` from the frozen dataset. The default comparison
is order-insensitive for parallel calls. Summary metrics count valid structure,
function names, argument names, argument values, and complete call matches.

## Outputs

For each evaluation directory:

- `scored_predictions.jsonl`: one scored row for every dataset record.
- `parse_failures.jsonl`: rows with malformed, missing, or errored output.
- `scores.json`: aggregate counts and rates.

Baseline generation uses the unmodified base model. Adapter evaluation loads the
base model in a fresh process and attaches the saved PEFT adapter before
generation.
