# EC2 Instance Policy

This policy records how Qwen3-1.7B experiment hosts should be selected and
operated. It is a cost-control rule for normal work, not a replacement for the
run manifest or artifact contract.

## Operating Policy

- Use local compute for dataset-only work. Dataset normalization, split
  selection, local scoring, manifest checks, report generation, and S3 inventory
  checks do not require a GPU unless the input data becomes too large for the
  workstation.
- Keep at most one GPU EC2 instance running at any time.
- Reuse the existing stopped instance for normal GPU work:
  `i-0c769a18f50fd1fe6`.
- Change the stopped instance's `InstanceType` before launch instead of
  launching separate task-specific instances.
- Keep the retained workspace EBS volume attached to the reused instance so
  model caches, Docker/NGC caches, datasets, checkpoints, and results remain
  available across starts and stops.
- Stop the instance after an approved cloud run unless the next approved task is
  immediately chained and requires the host to remain online.
- Do not terminate the instance, delete EBS volumes, or replace the retained
  workspace volume unless explicitly approved.

Current retained volumes:

| Purpose | Volume | Device | Retention |
| --- | --- | --- | --- |
| Root volume | `vol-0c4703d8ee8b9e1e7` | `/dev/sda1` | Delete on termination |
| Workspace volume | `vol-09a2fd5650d8c763e` | `/dev/sdf` | Retained |

## Task To Instance Map

| Task class | Default placement | Reason |
| --- | --- | --- |
| Dataset manipulation | Local workstation | No GPU needed. Avoid EC2 cost. |
| Larger dataset manipulation | Local workstation first | Use EC2 only if local RAM/disk becomes the blocker. |
| Base model inference | `g6.xlarge` | Fits Qwen3-1.7B inference with ample GPU memory. |
| LoRA adapter inference | `g6.xlarge` | Adapter inference has similar memory needs to base inference. |
| QLoRA adapter inference | `g6.xlarge` | Quantized adapter inference does not need the larger L40S host. |
| Full-parameter checkpoint inference | `g6.xlarge` first | Inference is much smaller than full-parameter training. Escalate only on OOM. |
| LoRA training | `g6.2xlarge` | More CPU/RAM headroom than `g6.xlarge`; GPU memory is still sufficient for smoke and expected LoRA runs. |
| QLoRA training | `g6.2xlarge` | Quantized training should fit on the L4 host while preserving enough host memory. |
| Full-parameter training smoke | `g6e.2xlarge` | Full model gradients/optimizer state need the larger L40S memory budget. |
| Full-parameter serious run | `g6e.2xlarge` or larger by evidence | Start from the proven L40S host; increase only if measured memory or throughput requires it. |

Escalation rule: start with the default instance above, record peak allocated and
reserved GPU memory in the run manifest, and move to a larger instance only when
there is measured OOM, memory pressure, or unacceptable throughput.

Task-specific exception: Experiment 3 / Task 07 reference BF16 LoRA starts on
`g6e.2xlarge` because the protocol explicitly requests that host and because the
run includes batch-size probing plus hook-level memory attribution. Use the
resulting peak reserved VRAM and throughput evidence to decide whether later
routine LoRA runs should move back to `g6.2xlarge`.

## Instance Type Switch Commands

The instance must be stopped before changing type.

```bash
export AWS_PROFILE=finetuning-local
export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION=us-west-2
export AWS_PAGER=""

aws ec2 describe-instances \
  --instance-ids i-0c769a18f50fd1fe6 \
  --query 'Reservations[0].Instances[0].{state:State.Name,type:InstanceType}' \
  --output table
```

Switch to the inference host:

```bash
aws ec2 modify-instance-attribute \
  --instance-id i-0c769a18f50fd1fe6 \
  --instance-type Value=g6.xlarge
```

Switch to the LoRA/QLoRA training host:

```bash
aws ec2 modify-instance-attribute \
  --instance-id i-0c769a18f50fd1fe6 \
  --instance-type Value=g6.2xlarge
```

Switch to the full-parameter training host:

```bash
aws ec2 modify-instance-attribute \
  --instance-id i-0c769a18f50fd1fe6 \
  --instance-type Value=g6e.2xlarge
```

Start and later stop the reused instance:

```bash
aws ec2 start-instances --instance-ids i-0c769a18f50fd1fe6
aws ec2 wait instance-running --instance-ids i-0c769a18f50fd1fe6

# Run the approved task, sync artifacts, and verify durable outputs first.

aws ec2 stop-instances --instance-ids i-0c769a18f50fd1fe6
aws ec2 wait instance-stopped --instance-ids i-0c769a18f50fd1fe6
```

## Launch Templates

The following templates exist as recovery/reference definitions:

| Template | Instance type | Normal use |
| --- | --- | --- |
| `ft-qwen17-infer-g6-xlarge-v1` | `g6.xlarge` | Reference only |
| `ft-qwen17-lora-g6-2xlarge-v1` | `g6.2xlarge` | Reference only |
| `ft-qwen17-full-g6e-2xlarge-v1` | `g6e.2xlarge` | Reference only |

Do not use these templates for normal task switching. Launching from a template
creates a new instance and new volumes, which defeats the shared-cache policy
unless a replacement-host procedure is explicitly approved.

## Cost Notes

Stopped EC2 instances do not accrue instance-hour charges, but retained EBS
volumes and snapshots still accrue storage charges. AWS instance prices change,
so verify current regional pricing before long runs or before changing this
policy.
