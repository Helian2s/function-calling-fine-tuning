# EBS Cleanup Execution

Executed on EC2 instance `i-0c769a18f50fd1fe6` in `us-west-2` on 2026-06-29 UTC.

## Commands

- Inventory command: `572efc6e-f2fb-4fc8-9f7e-a35fff43908b`
- Workspace inventory command: `f6e21a35-36d1-4ec3-9c54-c2461b0e4e04`
- Cleanup command: `4cd287ac-235f-4263-b16c-719a75358163`
- Final tidy command: `2f01380a-514a-4f02-9f86-1c619fd6403e`

## Before

- Retained workspace mount: `/mnt/workspace`
- Filesystem: `/dev/nvme1n1`, 246 GiB
- Used before cleanup: 190 GiB
- Free before cleanup: 43 GiB
- Major usage:
  - `/mnt/workspace/containerd`: 106 GiB
  - `/mnt/workspace/checkpoints`: 54 GiB
  - `/mnt/workspace/huggingface-cache`: 29 GiB

## Removed

- Fine-tuning checkpoints except `/mnt/workspace/checkpoints/exp-08/train-full`
- Fine-tuning results, logs, data, run-info, source bundles, source updates, backups, and temp payloads
- Unused Docker images and build state via `docker system prune -af`
- Old `/mnt/workspace/project` copy and empty Docker placeholder directories

## Retained

- `/mnt/workspace/huggingface-cache`: 29 GiB
- `/mnt/workspace/checkpoints/exp-08/train-full`: 150 MiB
- `/mnt/workspace/ngc-cache`: 4 KiB

## After

- Used after cleanup: 29 GiB
- Free after cleanup: 205 GiB
- Workspace use after cleanup: 13%
- Docker images after cleanup: 0

## Instance State

The instance did not complete normal shutdown promptly and remained in `stopping`.
It was then force-stopped with `aws ec2 stop-instances --force`, which stops but
does not terminate the instance. Final observed state: `stopped`.

## Cost Note

This cleanup frees filesystem space but does not reduce EBS provisioned-capacity
charges. The attached volumes are still 100 GiB root plus 250 GiB retained
workspace. Reducing EBS daily cost requires replacing, shrinking, or deleting
volumes in a separate storage-change step.
