"""Fetch a small TSRD sample for loader validation.

Downloads exactly 3 scan-mode train files from the Turing Synthetic Radar
Dataset on HuggingFace Hub into data/tsrd_sample/.

Prerequisites:
  pip install "ew-smart-scan[data]"      # installs huggingface_hub
  export HF_TOKEN=<your_huggingface_token>

The dataset is GATED — you must:
  1. Create a HuggingFace account
  2. Visit https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
  3. Accept the dataset access agreement
  4. Generate an access token at https://huggingface.co/settings/tokens

Usage:
  python scripts/fetch_tsrd_sample.py [--n-files 3] [--out data/tsrd_sample]
  python -m scripts.fetch_tsrd_sample

Repo structure (confirmed by live listing, no assumption):
  scan/train_scan/config_NNN.h5   — scan-mode, train split (2500 files)
  scan/val_scan/config_NNN.h5     — scan-mode, validation split (250 files)
  scan/test_scan/config_NNN.h5    — scan-mode, test split (250 files)
  stare/train_stare/config_NNN.h5 — stare-mode (very large, ~1.27M pulses each)
  ...

This script only downloads scan/train_scan/ files (safe, ~94K pulses each).
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)

REPO_ID   = "alan-turing-institute/turing-synthetic-radar-dataset"
REPO_TYPE = "dataset"

# Confirmed path pattern from live HuggingFace listing (Sep 2026)
SCAN_TRAIN_PREFIX = "scan/train_scan/"


def _require_huggingface_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        print(
            "huggingface_hub is not installed.\n"
            "Install the data extra:\n"
            "  pip install \"ew-smart-scan[data]\"\n"
            "or directly:\n"
            "  pip install huggingface_hub==0.24.6",
            file=sys.stderr,
        )
        sys.exit(1)


def _get_token() -> str | None:
    """Return HF_TOKEN from environment, or None if unset (unauthenticated)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN not set. The TSRD dataset is gated — unauthenticated "
            "downloads will fail with 401. Set HF_TOKEN and ensure you have "
            "accepted the dataset access agreement at:\n"
            "  https://huggingface.co/datasets/%s",
            REPO_ID,
        )
    # Deliberately do NOT log or print the token value
    return token or None


def list_scan_train_files(hf, token: str | None) -> list[str]:
    """List all scan/train_scan/*.h5 files in the HF repo.

    Prints the first 20 raw filenames BEFORE filtering so the actual path
    pattern is visible in the output — per task spec.
    """
    api = hf.HfApi()

    print(f"Listing files in {REPO_ID} ...")
    all_files = list(api.list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
    ))

    print(f"\nTotal files in repo: {len(all_files)}")
    print("First 20 raw filenames (before filtering):")
    for f in all_files[:20]:
        print(f"  {f}")

    # Filter: scan-mode, train split, .h5 extension
    scan_train = [
        f for f in all_files
        if f.startswith(SCAN_TRAIN_PREFIX) and f.endswith(".h5")
    ]

    print(f"\nAfter filtering for '{SCAN_TRAIN_PREFIX}*.h5': {len(scan_train)} files found")
    if not scan_train:
        print(
            "WARNING: filter returned 0 files. Check that the path prefix is correct.\n"
            "Known prefixes in this repo:\n"
            "  scan/train_scan/  (2500 files)\n"
            "  scan/val_scan/    (250 files)\n"
            "  stare/train_stare/ (2500 files, very large)"
        )
    return sorted(scan_train)


def download_sample(
    out_dir: pathlib.Path,
    n_files: int = 3,
    token: str | None = None,
) -> list[pathlib.Path]:
    """Download n_files scan-mode train files into out_dir.

    Returns list of local paths.
    """
    hf = _require_huggingface_hub()

    scan_train_files = list_scan_train_files(hf, token)
    if not scan_train_files:
        print("No files matched — nothing downloaded.", file=sys.stderr)
        sys.exit(1)

    to_download = scan_train_files[:n_files]
    print(f"\nDownloading {len(to_download)} file(s) to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[pathlib.Path] = []
    for repo_path in to_download:
        filename = pathlib.Path(repo_path).name
        local_path = out_dir / filename
        print(f"  {repo_path} → {local_path}")
        try:
            dl = hf.hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type=REPO_TYPE,
                local_dir=str(out_dir),
                token=token,
            )
            # hf_hub_download may put the file in a subdirectory matching the
            # repo path — find the actual location
            downloaded = pathlib.Path(dl)
            if downloaded.exists():
                local_paths.append(downloaded)
                print(f"    ✓ saved to {downloaded}")
            else:
                print(f"    ✗ expected file not found at {downloaded}", file=sys.stderr)
        except Exception as exc:
            print(f"    ✗ failed: {exc}", file=sys.stderr)
            if "GatedRepoError" in type(exc).__name__ or "401" in str(exc):
                print(
                    "\n    The dataset is GATED. You must:\n"
                    "    1. Visit https://huggingface.co/datasets/"
                    f"{REPO_ID}\n"
                    "    2. Accept the access agreement\n"
                    "    3. Set HF_TOKEN to a token with 'read' permissions\n",
                    file=sys.stderr,
                )
                sys.exit(1)

    return local_paths


def main(n_files: int = 3, out_dir: pathlib.Path = pathlib.Path("data/tsrd_sample")) -> list[pathlib.Path]:
    token = _get_token()
    local_paths = download_sample(out_dir=out_dir, n_files=n_files, token=token)

    print("\n" + "─" * 60)
    print("Downloaded files:")
    for p in local_paths:
        print(f"  {p.resolve()}")
    print("─" * 60)
    print("\nNext: run the loader validation test:")
    print("  python3 -m pytest tests/test_tsrd_real_data.py -v -s")
    return local_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-files", type=int, default=3,
                        help="Number of files to download (default: 3)")
    parser.add_argument("--out", default="data/tsrd_sample",
                        help="Output directory (default: data/tsrd_sample)")
    args = parser.parse_args()
    main(n_files=args.n_files, out_dir=pathlib.Path(args.out))
