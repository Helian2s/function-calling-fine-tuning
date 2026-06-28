# Experiment 0 Completion

- Status: `complete`
- May proceed: `True`
- Generated at: `2026-06-26T16:37:14.575912+00:00`

## Stage Status

| Stage | Status | Summary |
| --- | --- | --- |
| gpu_driver_docker_cuda_mount_s3_preflight | `pass` | environment/package evidence is present |
| qwen3_1_7b_pinned_load | `pass` | generation metadata records pinned Qwen3-1.7B load |
| native_template_rendering | `pass` | at least five native-template examples rendered with thinking disabled |
| loss_mask | `pass` | loss includes assistant tool-call spans only |
| untouched_base_generation_40 | `pass` | 40 predictions verified against stored scores |
| smoke_training_30_steps | `pass` | 30-step training has finite loss and adapter-only evidence |
| adapter_save | `pass` | adapter config and weights exist in retained storage |
| clean_process_reload | `pass` | adapter reload report is deterministic |
| post_training_generation_parse_score_40 | `pass` | 40 predictions verified against stored scores |
| peak_allocated_reserved_vram | `pass` | peak allocated/reserved VRAM is recorded for all GPU stages |
| canonical_artifact_bundle | `pass` | all canonical final result files are present |
| s3_artifact_upload | `pass` | required S3 artifact names are present in inventory |
| ec2_instance_stopped | `pass` | EC2 instance is stopped |

## Blockers

- None
