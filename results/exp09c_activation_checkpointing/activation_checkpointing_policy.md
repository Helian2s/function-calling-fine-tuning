# Exp09C Activation Checkpointing Benchmark

Status: completed with failed activation-checkpointing profile.

## Result

- LoRA off: completed 300 observed training steps with controlled monitor stop.
- LoRA on: failed before the first training step.
- Failure: Activation checkpointing failed before the first training step in the pinned NeMo AutoModel BF16 LoRA path: checkpointed inputs had no requires_grad=True, so backward failed with no grad_fn.

## Key Metrics

| Profile | Status | Step | Duration s | Max GPU MB | Return |
|---|---:|---:|---:|---:|---:|
| lora_off | controlled stop | 299 | 221.072 | 38663.0 | 0 |
| lora_on | failed before step | None | 20.093 | 4391.0 | 1 |

## Policy

- PEFT on L40S: `off`
- PEFT on L4: `off`
- Reason: activation-checkpointing profile failed under the pinned BF16 LoRA path
- Packing interaction measured: `false`

Full metrics and logs are in S3 under `finetuning/results/exp-09c/` and `finetuning/logs/exp-09c/`.

Note: hook-level allocated/reserved VRAM for `lora_off` is unavailable because the monitor intentionally stopped the process after step 299; the reported control memory signal is GPU-log max memory.
