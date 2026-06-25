# Experiment 0 AWS Infrastructure

This repository prepares artifacts for a controlled GPU smoke run. It must not
launch, stop, terminate, or modify EC2 instances unless the operator explicitly
requests that action after reviewing the C0 audit.

## Bootstrap Artifacts

- `infrastructure/aws/bootstrap/bootstrap_instance.sh` is the EC2 user-data
  bootstrap payload. It mounts the retained workspace EBS volume, downloads the
  release manifest, verifies the immutable source archive checksum, verifies the
  frozen dataset checksums, and installs the shutdown/sync helper.
- `infrastructure/aws/bootstrap/shutdown_and_sync.sh` is installed on the host
  as `/usr/local/sbin/ft-exp00-shutdown-and-sync`. It requires explicit
  invocation, supports `--dry-run`, uploads results/logs/run-info, uploads the
  final adapter for final-stage runs, and calls `shutdown -h now`. It does not
  upload base model caches, Docker/NGC caches, or intermediate checkpoints.

## Local Preparation

Build the source bundle only from a clean committed tree:

```bash
scripts/build_exp00_source_bundle.sh
```

Preview publication without writing to S3:

```bash
scripts/publish_exp00_source_bundle.sh --dry-run
```

Publish the current clean Git commit before each EC2 experiment stage:

```bash
scripts/sync_source_to_s3.sh --execute
```

The frozen smoke dataset stays canonical in S3 and is copied to
`/mnt/workspace/data` during bootstrap. The current smoke-v1 prefix is about
454 KiB, so this copy is intentionally cheap and deterministic.

Preview post-run artifact upload before stopping an instance:

```bash
scripts/sync_results.sh --stage baseline --dry-run
scripts/sync_results.sh --stage final --dry-run
```

Audit the launch template with read-only AWS CLI calls:

```bash
scripts/audit_launch_template.sh
```

The publication and audit scripts use profile `finetuning-local` and region
`us-west-2` by default. The audit script does not launch an instance.
