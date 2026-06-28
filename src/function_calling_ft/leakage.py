from __future__ import annotations

from pathlib import Path
from typing import Any

from function_calling_ft.curation import (
    audit_group_leakage,
    read_jsonl,
    write_json,
)


def run_leakage_audit(
    *,
    group_metadata_path: Path,
    output_path: Path,
    fail_on_overlap: bool = True,
) -> dict[str, Any]:
    report = audit_group_leakage(read_jsonl(group_metadata_path))
    write_json(output_path, report)

    if fail_on_overlap and report["status"] != "pass":
        raise RuntimeError(
            "Leakage audit failed: split groups appear in multiple splits."
        )

    return report
