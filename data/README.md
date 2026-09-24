# Data

This project uses statistics extracted from the Turing Synthetic Radar Dataset (TSRD).

TSRD is available at: https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset

---

## Download Instructions

To reproduce the real-data evaluation:

1. Obtain a HuggingFace token with read access at https://huggingface.co/settings/tokens

2. Set the token as an environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. Install the data download dependency:
   ```bash
   pip install huggingface_hub>=0.24
   ```

4. Download the TSRD sample files:
   ```bash
   python scripts/fetch_tsrd_sample.py
   ```
   This places `.h5` files in `data/tsrd_sample/scan/train_scan/`.

5. Run the full evaluation harness:
   ```bash
   bash scripts/run_evaluation.sh
   ```

---

## What's Committed vs. What's Downloaded

| Item | Status | Reason |
|---|---|---|
| `data/README.md` | Committed | Instructions |
| `data/.gitkeep` | Committed | Placeholder |
| `data/tsrd_sample/*.h5` | **Not committed** | Large files (100MB+), gated license |
| `data/real_data_eval_results.md` | Committed | Summary of real-data evaluation |
| `data/ablation_module_c_results.md` | Committed | Module C ablation results |

The TSRD `.h5` files are not committed to this repository. They must be downloaded separately using the instructions above.

---

## TSRD File Schema

Each `.h5` file contains:
- `data["data"]`: shape (n_pulses, 5) → columns: [ToA (μs), CF (MHz), PW (μs), AoA (deg), Amplitude (dB)]
- `data["labels"]`: shape (n_pulses,) → integer emitter IDs (file-local)

ToA values in real TSRD are non-monotonic (interleaved pulses across emitters).

---

## TSRD-Grounded Parameters Used in This Project

| Emitter Class | Period T (μs) | Jitter σ (μs) | CoV | Notes |
|---|---|---|---|---|
| P1-TSRD | 1511.5 | 133.2 | 8.8% | Emitter 10, config_0.h5 |
| P2-TSRD | 502.5 | 35.9 | 7.1% | Emitter 31, config_2.h5 |

**Note:** Only 2 periodic emitter instances were characterised from 3 TSRD files. 95% of TSRD emitters (125 total) have CoV ≥ 1.0 and are not periodic in the classical fixed-PRI sense. See `docs/limitations.md` §5 for details.
