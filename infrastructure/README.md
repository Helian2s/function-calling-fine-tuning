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
  invocation, supports `--dry-run`, uploads results/checkpoints/logs/run-info,
  and calls `shutdown -h now`.

## Local Preparation

Build the source bundle only from a clean committed tree:

```bash
scripts/build_exp00_source_bundle.sh
```

Preview publication without writing to S3:

```bash
scripts/publish_exp00_source_bundle.sh --dry-run
```

Audit the launch template with read-only AWS CLI calls:

```bash
scripts/audit_launch_template.sh
```

The publication and audit scripts use profile `finetuning-local` and region
`us-west-2` by default. The audit script does not launch an instance.
