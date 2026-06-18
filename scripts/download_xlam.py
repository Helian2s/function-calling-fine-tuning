from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "Salesforce/xlam-function-calling-60k"
FILENAME = "xlam_function_calling_60k.json"

RAW_DIR = Path("data/raw/xlam")
MANIFEST_PATH = Path("data/manifests/xlam_source.json")


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 without loading the entire file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def card_license(card_data: Any) -> str | None:
    """Safely read the license value from Hugging Face card metadata."""
    if card_data is None:
        return None

    return getattr(card_data, "license", None)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    info = api.dataset_info(
        REPO_ID,
        token=True,
        files_metadata=True,
    )

    revision = info.sha
    if not revision:
        raise RuntimeError("Hugging Face did not return a repository revision.")

    downloaded_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            revision=revision,
            local_dir=RAW_DIR,
            token=True,
        )
    )

    if not downloaded_path.is_file():
        raise FileNotFoundError(
            f"Downloaded dataset file was not found: {downloaded_path}"
        )

    manifest = {
        "repository_id": REPO_ID,
        "repository_type": "dataset",
        "filename": FILENAME,
        "revision": revision,
        "license": card_license(info.card_data),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(downloaded_path),
        "size_bytes": downloaded_path.stat().st_size,
        "sha256": calculate_sha256(downloaded_path),
        "access": "gated",
        "raw_data_committed_to_git": False,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Downloaded: {downloaded_path}")
    print(f"Revision:   {revision}")
    print(f"SHA-256:   {manifest['sha256']}")
    print(f"Manifest:  {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
