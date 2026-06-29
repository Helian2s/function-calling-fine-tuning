# Fine-Tuning Artifact Retention

## Policy

Git stores knowledge and reproducibility metadata; S3/EBS store only artifacts needed for immediate next tasks.

## Keep In Git

- `docs/fine_tuning_final_report.md`
- `docs/fine_tuning_artifact_retention.md`
- `results/fine_tuning_closure/final_metric_table.csv`
- `results/fine_tuning_closure/final_metric_table.json`
- `results/fine_tuning_closure/experiment_decisions.json`
- `results/fine_tuning_closure/decision_artifacts/*.json`
- `results/fine_tuning_closure/artifact_inventory.json`
- `results/fine_tuning_closure/checksums.sha256`

## Keep Outside Git Temporarily

- S3 `finetuning/checkpoints/exp-08/train-full/`: Small best LoRA adapter from the full-data run; optional TensorRT-LLM deployment comparison.
- EBS `Hugging Face cache for Qwen/Qwen3-1.7B if present`: Speeds up TensorRT-LLM benchmark setup; reproducible from Hugging Face if absent.
- EBS `best LoRA adapter if present`: Optional future fine-tuned deployment benchmark.

## Cleanup Candidates

- Full-SFT checkpoints and optimizer state
- Pilot checkpoints
- Intermediate LoRA/QLoRA/rank/target checkpoints
- Repeated source/workspace bundles
- Temporary Curator payloads
- S3 noncurrent versions for deleted artifacts
- EBS result/log/checkpoint duplicates after git closure is committed

## Cost Note

S3 cleanup reduces object storage cost directly. EBS filesystem cleanup frees space but does not reduce gp3 volume charges unless the volume is resized, replaced, or deleted.
