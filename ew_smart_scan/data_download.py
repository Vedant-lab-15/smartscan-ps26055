"""TSRD dataset download helper.

Downloads Turing Synthetic Radar Dataset files from HuggingFace Hub.

HF_TOKEN must be set as an environment variable — never hardcoded.
Raises CredentialError if the token is missing.

Usage:
    python -m ew_smart_scan.data_download --subset scan --split train --out data/

Or from code:
    from ew_smart_scan.data_download import download_tsrd
    download_tsrd(target_dir=Path("data"), subsets=["train"], modes=["scan"])

Dataset: alan-turing-institute/turing-synthetic-radar-dataset
Cite: Gunn et al. (2026), arXiv:2602.03856
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib

from ew_smart_scan.env.errors import CredentialError

logger = logging.getLogger(__name__)

HF_DATASET_REPO = "alan-turing-institute/turing-synthetic-radar-dataset"

# Available splits and receiver modes
VALID_SUBSETS = ("train", "validation", "test")
VALID_MODES   = ("scan", "stare")


def _get_token() -> str:
    """Read HF_TOKEN from environment. Raises CredentialError if absent."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise CredentialError(
            "HF_TOKEN environment variable is not set.\n"
            "Generate a token at https://huggingface.co/settings/tokens and export it:\n"
            "  export HF_TOKEN=<your_token>\n"
            "Never hardcode the token in source files or commit it to version control."
        )
    # Deliberately do NOT log or print the token value
    return token


def download_tsrd(
    target_dir: pathlib.Path,
    subsets: list[str] | None = None,
    modes: list[str] | None = None,
) -> pathlib.Path:
    """Download TSRD files from HuggingFace Hub to target_dir.

    Args:
        target_dir: Local directory to save files into. Created if absent.
        subsets:    Which data splits to download. Defaults to ["train"].
                    Options: "train", "validation", "test".
        modes:      Receiver modes to download. Defaults to ["scan"].
                    Options: "scan", "stare".
                    WARNING: stare-mode files are very large (~1.27M pulses
                    each, ~3.9B total). Download sparingly — use scan for
                    development, stare only for oracle evaluation.

    Returns:
        Path to the root download directory.

    Raises:
        CredentialError: if HF_TOKEN is not set.
        ImportError: if huggingface_hub is not installed.
    """
    # Credential check happens FIRST — before any heavy imports
    token = _get_token()  # raises CredentialError if HF_TOKEN missing

    try:
        from turing_deinterleaving_challenge import download_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "turing-deinterleaving-challenge is not installed.\n"
            "Install with:\n"
            "  pip install git+https://github.com/alan-turing-institute/"
            "turing-deinterleaving-challenge.git"
        ) from exc

    subsets = subsets or ["train"]
    modes   = modes   or ["scan"]

    # Validate inputs
    for s in subsets:
        if s not in VALID_SUBSETS:
            raise ValueError(f"Invalid subset '{s}'. Choose from {VALID_SUBSETS}.")
    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"Invalid mode '{m}'. Choose from {VALID_MODES}.")

    if "stare" in modes:
        logger.warning(
            "Downloading stare-mode data. Stare files average ~1.27M pulses each "
            "and ~3.9B pulses total across the full split. Download only what you need "
            "and never load the full stare split into memory at once."
        )

    token = _get_token()  # raises CredentialError if missing; token not logged

    target_dir = pathlib.Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading TSRD subsets=%s, modes=%s → %s", subsets, modes, target_dir
    )

    # Use the official download_dataset function from the challenge package.
    # It handles resumable downloads and HuggingFace authentication internally.
    dataset_path = download_dataset(
        save_dir=target_dir,
        subsets=subsets,
        modes=modes,
        token=token,
    )

    logger.info("Download complete: %s", dataset_path)
    return pathlib.Path(dataset_path)


def list_h5_files(
    data_dir: pathlib.Path,
    subset: str = "train",
    mode: str = "scan",
    max_files: int | None = None,
) -> list[pathlib.Path]:
    """Return sorted list of .h5 files for a given subset and receiver mode.

    Args:
        data_dir:  Root data directory (where download_tsrd saved files).
        subset:    "train", "validation", or "test".
        mode:      "scan" or "stare".
        max_files: If set, return at most this many files (useful for dev runs).

    Returns:
        Sorted list of pathlib.Path objects pointing to .h5 files.
    """
    pattern_dir = pathlib.Path(data_dir) / subset / mode
    if not pattern_dir.exists():
        # Try flat layout (some versions of the package use a flat dir)
        pattern_dir = pathlib.Path(data_dir) / f"{subset}_{mode}"
    if not pattern_dir.exists():
        logger.warning("Data directory not found: %s", pattern_dir)
        return []

    files = sorted(pattern_dir.glob("*.h5"))
    if max_files is not None:
        files = files[:max_files]

    logger.info("Found %d .h5 files in %s", len(files), pattern_dir)
    return files


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Turing Synthetic Radar Dataset from HuggingFace."
    )
    p.add_argument(
        "--subset", nargs="+", default=["train"],
        choices=list(VALID_SUBSETS),
        help="Data splits to download (default: train).",
    )
    p.add_argument(
        "--mode", nargs="+", default=["scan"],
        choices=list(VALID_MODES),
        help="Receiver modes to download (default: scan). "
             "WARNING: stare is very large.",
    )
    p.add_argument(
        "--out", default="data",
        help="Local directory to save data (default: data/).",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    download_tsrd(
        target_dir=pathlib.Path(args.out),
        subsets=args.subset,
        modes=args.mode,
    )
