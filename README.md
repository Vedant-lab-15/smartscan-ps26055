# EW Smart Scan Strategy

**SIH 2026 — PS 26055 (DRDO)**

Research-grounded prototype for Electronic Warfare receiver scheduling using
Restless Multi-Armed Bandit / POMDP theory and Whittle index schedulers.
Not a claim of fielded deployment.

---

## Architecture Overview

The system schedules K simultaneous frequency-band scans out of N total bands using:

- **Module A** — WIQL-UCB baseline → Neural-Q-Whittle scheduler
- **Module B** — DQWIC contextual threat-weighting layer
- **Module C** — Renewal-process Whittle index for periodic emitters
- **Module D** — Adversarial jitter robustness via Kalman filter tracking

Training and evaluation use the **Turing Synthetic Radar Dataset (TSRD)**
(Gunn et al., 2026).

---

## Installation

### 1. Clone and install the package

```bash
pip install -e ".[dev]"
```

### 2. Install PyTorch (CPU-only)

PyTorch is a heavy dependency and is NOT included in `pyproject.toml`.
Install it separately:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

Or use the pinned file:

```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the Turing Deinterleaving Challenge package

This is a git-only package and is not on PyPI:

```bash
pip install git+https://github.com/alan-turing-institute/turing-deinterleaving-challenge.git
```

---

## HuggingFace Token Setup

TSRD data is hosted on HuggingFace Hub. You must set the `HF_TOKEN` environment
variable before downloading:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

**Never hardcode your token in source files or commit it to version control.**

The system will raise a `CredentialError` if `HF_TOKEN` is not set when a
dataset download is attempted.

---

## Data

Place TSRD `.h5` files in the `data/` directory. The `data/.gitkeep` file
is a placeholder — actual data files are git-ignored.

---

## Running Tests

```bash
pytest tests/ -q
```

---

## Running the Smoke Test

```bash
python scripts/smoke_test.py
```

> **Note**: The smoke test is a stub until Task 6 is implemented.
