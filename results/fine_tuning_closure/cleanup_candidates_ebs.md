# EBS Cleanup Plan

The existing stopped instance keeps two attached gp3 volumes:

- root volume: 100 GiB, delete-on-termination true
- retained workspace volume: 250 GiB, delete-on-termination false

Cleaning files from EBS frees filesystem space but does not reduce the EBS bill
unless the volume is shrunk, replaced, or deleted. Because the next TensorRT-LLM
task can reuse the same 250 GiB workspace, the recommended action is filesystem
cleanup only, followed by a later storage resize decision if needed.

## Keep

- Hugging Face cache for `Qwen/Qwen3-1.7B` if present.
- Best LoRA adapter if present and small.
- Shell history is not collected or preserved.

## Remove

- `/workspace/checkpoints` except the best full-data LoRA adapter if present.
- `/workspace/results`
- `/workspace/logs`
- `/workspace/source-bundles`
- `/workspace/source-updates`
- `/workspace/tmp`
- duplicated fine-tuning datasets under `/workspace/data`
- Docker build cache and stopped containers if present.

## Required Postconditions

- EC2 instance is stopped.
- Workspace free-space report is captured.
- No secrets are printed or persisted.
