# Controlled Demonstration Scripts

These scripts produce results for **bespoke experimental configurations** that are not part of the main evaluation harness. They require TSRD data (`.h5` files) and produce outputs in `results/controlled_demonstrations/`.

---

## standalone_0975_demonstration.py

**What it does:** Runs Module C (periodic interceptor) in an idealized standalone setup — single emitter, dedicated receiver, no competition for scan slots. Produces the 0.975 intercept rate cited in the paper as a controlled demonstration.

**Dependencies:** TSRD `.h5` files in `data/tsrd_sample/` (download via `scripts/fetch_tsrd_sample.py`)

**Run:**
```bash
python scripts/controlled_demonstrations/standalone_0975_demonstration.py
```

**Output:** Results printed to stdout. CSVs in `results/controlled_demonstrations/`.

**Why this is separate:** The scenario (single emitter, 2000 steps, dt=50μs) differs from the standard harness (multi-emitter, 5000 steps, dt=100/1000μs). This is a demonstration of maximum achievable performance, not an operational benchmark.
