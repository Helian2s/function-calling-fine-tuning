# Fine-Tuning Final Report

Generated: 2026-06-29T00:29:32.836161+00:00

## Scope

This report closes the Qwen3-1.7B fine-tuning workstream. The next workstream is model optimization and GPU acceleration, so this repository now preserves conclusions and reproducibility metadata rather than bulky training artifacts.

## Main Decisions

- Use the base `Qwen/Qwen3-1.7B` model first for the TensorRT-LLM optimization benchmark.
- Preserve only the small best LoRA adapter as an optional future deployment comparison.
- Do not preserve full-SFT checkpoints by default; the checkpoint is expensive and not needed for the next two projects.
- Keep BF16 deterministic generation as the primary comparable evaluation policy.
- Treat no-tool behavior as a known regression area for tool-call fine-tuning.
- Do not use activation checkpointing with the pinned BF16 LoRA path; it failed before the first step.

## Best Validation Rows

| experiment | variant | exec complete | complete-call F1 | arg value accuracy | no-tool FP | notes |
|---|---:|---:|---:|---:|---:|---|
| exp-08 | train_full | 0.8074 | 0.8515 | 0.9294 |  | One epoch per dataset size. |
| exp-06 | lora_rank4_alpha8_attention | 0.7880 | 0.8343 | 0.9168 |  | Full validation rank sweep. |
| exp-07 | lora_attention | 0.7880 | 0.8343 | 0.9168 |  | Attention-only versus attention+MLP target placement. |
| exp-08 | train_10k | 0.7880 | 0.8343 | 0.9168 |  | One epoch per dataset size. |
| exp-07 | lora_attention_mlp | 0.7806 | 0.8273 | 0.9150 |  | Attention-only versus attention+MLP target placement. |
| exp-06 | lora_rank8_alpha16_attention | 0.7634 | 0.8172 | 0.9122 |  | Full validation rank sweep. |
| exp-08 | train_2k | 0.7618 | 0.8102 | 0.9071 |  | One epoch per dataset size. |
| exp-06 | lora_rank16_alpha32_attention | 0.7325 | 0.7881 | 0.8939 |  | Full validation rank sweep. |

## Experiment Conclusions

- Base BF16 deterministic inference is the clean baseline for future optimization.
- NF4 inference reduced memory but hurt quality enough that it was not accepted as the primary comparison mode.
- BF16 LoRA improved tool-call accuracy over base, but caused severe no-tool false positives.
- QLoRA produced similar tool-call quality to reference LoRA under controlled settings, with stack-specific deviations documented.
- Full-parameter SFT improved tool-call metrics over PEFT on the 1K development comparison but was operationally much more expensive.
- Rank 4 / alpha 8 attention-only was selected as the smallest adequate LoRA configuration on full validation.
- Adding MLP adapters did not improve validation quality enough to justify the extra cost and worsened no-tool behavior.
- The full training pool produced the best tool-call validation scores, but no candidate satisfied all no-tool guardrails.
- Assistant-only masking remains the production policy; full-sequence loss was diagnostic only.
- Activation checkpointing failed in the pinned BF16 LoRA path because checkpointed inputs had no `requires_grad=True`.

## Durable Artifacts

- `results/fine_tuning_closure/final_metric_table.csv`
- `results/fine_tuning_closure/final_metric_table.json`
- `results/fine_tuning_closure/experiment_decisions.json`
- `results/fine_tuning_closure/decision_artifacts/`
- `results/fine_tuning_closure/artifact_inventory.json`
- `results/fine_tuning_closure/retained_artifacts_manifest.json`

## Cleanup Decision

S3 and EBS cleanup should remove checkpoints, optimizer states, repeated source bundles, temporary payloads, and raw run duplicates after this closure package is committed. The only optional model artifact retained outside git is the full-data LoRA adapter.

## Next Workstreams

1. TensorRT-LLM inference optimization benchmark on base Qwen3-1.7B.
2. Megatron/Megatron Core multi-GPU training benchmark using a synthetic small GPT-style setup.
