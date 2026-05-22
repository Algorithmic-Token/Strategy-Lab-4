# Strategy Lab #4 — Regime-Conditioned Opening Range Breakout

**Algorithmic Token · ENTER Invest**

> **Primary academic sources:**
> Mesfin, M. (2026) — *Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures: A Systematic Falsification Study*
> [arXiv:2605.04004](https://arxiv.org/abs/2605.04004) · q-fin.TR · May 2026

> Garg (2025) — *Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals*
> [arXiv:2512.12924](https://arxiv.org/abs/2512.12924) · q-fin.TR

> **Methodology reference:**
> López de Prado, M. (2018) — *Advances in Financial Machine Learning*, Chapter 17
> [Wiley](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)

Experimental algorithm implementation accompanying the Strategy Lab #4 article published at [Algorithmic Token on Substack](https://algorithmictoken.substack.com/p/strategy-lab-4-does-regime-conditioning).

---

## What This Is

Strategy Lab #4 is the direct continuation of Lab #3. Where Lab #3 falsified fourteen OHLCV signal families unconditionally, Lab #4 tests **Direction 1** from that study: does applying a volatility and volume regime filter to the Opening Range Breakout produce a signal that passes the five-criterion falsification harness?

The module extends `strategy_lab_03.py` with:

1. **A two-condition regime classifier** — identifies bars where both realised volatility and relative volume are elevated, based on the conditional applicability finding in Garg (2025)
2. **A regime sensitivity analysis** — maps filter aggressiveness against trade count to identify which threshold combinations satisfy Criterion 3 (minimum 30 trades per window) before running the full harness
3. **A side-by-side comparison** — unconditional ORB (Lab #3 baseline, expected to fail) versus regime-conditioned ORB (Lab #4 test), run through identical harness parameters for a clean verdict

The regime thresholds (60th volatility percentile, 1.10x relative volume) are applied as published in Garg (2025) — never fitted to test data.

---

## Repository Structure

```
strategy_lab_04/
├── strategy_lab_04.py   — regime classifier + conditioned ORB + harness
└── README.md            — this file
```

This module also depends on `strategy_lab_03/strategy_lab_03.py` for the falsification harness. In the monorepo, both folders must be present.

---

## Environment Setup and Installation

### Prerequisites

- Python 3.9 or higher
- `strategy_lab_03/` present in the same repository root

### Step 1 — Clone the monorepo

```bash
git clone https://github.com/Algorithmic-Token/Algorithmic-Token.git
cd Algorithmic-Token
```

### Step 2 — Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas yfinance scipy
```

### Step 4 — Verify installation

```bash
python3 -c "import numpy, pandas, yfinance, scipy; print('All dependencies OK')"
```

### Step 5 — Run the demo

```bash
python3 strategy_lab_04/strategy_lab_04.py
```

**Expected output:**
```
============================================================
Strategy Lab #4 — Regime-Conditioned ORB
Algorithmic Token · ENTER Invest
============================================================

Data loaded    : 4680 bars, 60 trading days
Regime active  : 24.3% of bars
Trade reduction: ~75.7% of ORB trades filtered

── Unconditional ORB (baseline, no regime filter) ──
  Window 01 [FAIL]  Trades= 14 | T= 0.71 | Net=$  -38.0  — T=0.71<2.0
  ...
  ── OVERALL VERDICT : FAIL ──

── Regime-Conditioned ORB ──
  Window 01 [FAIL]  Trades=  4 | T= 1.23 | Net=$  +12.0  — Trades=4<10
  ...
  ── OVERALL VERDICT : FAIL ──

── Regime Sensitivity (active bar % by threshold) ──
 vol_percentile  volume_ratio_min  active_bars_pct  approx_trades_per_63d_window
            0.5              1.05             45.0                          85.0
  ...
```

The ORB is expected to fail in the 60-day demo — insufficient data for meaningful walk-forward windows. Full replication requires 947 trading days from a paid intraday data vendor.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `vol_percentile` | `0.60` | Volatility activation threshold — 60th percentile of rolling realised vol |
| `volume_ratio_min` | `1.10` | Minimum relative volume — 10% above 60-day rolling average |
| `round_trip_cost` | `2.0` | Round-trip transaction cost in index points (MNQ: $4/contract) |
| `formation_days` | `126` | Walk-forward training window (use 15 for 60-day demo) |
| `test_days` | `63` | Walk-forward test window (use 10 for 60-day demo) |
| `min_trades` | `30` | Criterion 3 — minimum trades per window (use 10 for demo) |
| `min_tstat` | `2.0` | Criterion 2 — minimum T-statistic |
| `stability_threshold` | `0.75` | Criterion 5 — minimum pass rate across windows |

---

## The Five Institutional Criteria (from Lab #3)

| Criterion | Threshold | Description |
|---|---|---|
| **C1** Walk-forward | 126d / 63d | Train on formation, test out-of-sample — no look-ahead |
| **C2** T-statistic | ≥ 2.0 | Mean trade P&L statistically distinguishable from zero |
| **C3** Trade count | ≥ 30 | Minimum trades per window for T-stat reliability |
| **C4** Net return | > 0 | Positive P&L after full round-trip transaction costs |
| **C5** Stability | ≥ 75% | Must pass C2–C4 in at least 75% of all windows |

---

## Regime Sensitivity Reference

The sensitivity analysis maps filter aggressiveness against trade count before running the full harness. Tightening the regime filter below the minimum trade count threshold fails Criterion 3 regardless of signal quality.

| Vol Percentile | Volume Ratio | Active Bars % | Trades / Window | C3 Status |
|---|---|---|---|---|
| 50th | 1.05x | ~45% | ~85 | PASS |
| 50th | 1.10x | ~38% | ~72 | PASS |
| 50th | 1.20x | ~28% | ~53 | PASS |
| 60th | 1.05x | ~32% | ~60 | PASS |
| **60th** | **1.10x** | **~26%** | **~49** | **PASS — published** |
| 60th | 1.20x | ~18% | ~34 | MARGINAL |
| 70th | 1.05x | ~22% | ~41 | MARGINAL |
| 70th | 1.10x | ~16% | ~30 | FAIL |
| 70th | 1.20x | ~10% | ~19 | FAIL |

---

## Three Possible Verdicts

| Verdict | Meaning | Next Step |
|---|---|---|
| **Full PASS** | Regime conditioning rescues the ORB | Test Direction 2 — signal combination |
| **Partial PASS** | Edge in some windows, not stable | Refine regime definition — Direction 1 extended |
| **Full FAIL** | No edge even in favourable regimes | Move to Direction 3 — tick-level data |

---

## Data Limitation

**yfinance provides a maximum of 60 days of 5-minute intraday data.** The demo runs with reduced window parameters (`formation_days=15`, `test_days=10`, `min_trades=10`) to produce output on this limited dataset. For a meaningful falsification study use full institutional parameters on a paid data vendor.

| Vendor | Notes |
|---|---|
| **Interactive Brokers API** | Requires IB account; use `ib_insync` library |
| **Norgate Data** | Clean US futures data; Windows only |
| **Refinitiv / LSEG** | Institutional grade; comprehensive |
| **Alpaca Markets API** | Free tier for US equities; futures require subscription |

---

## Relationship to Other Strategy Labs

| Lab | Signal | Key Addition |
|---|---|---|
| Lab #1 | Momentum | Proportional-control vol targeting |
| Lab #2 | FX Mean Reversion | Cointegration + Q-learning RL agent |
| Lab #3 | Falsification Harness | Five institutional criteria, 14 signal families |
| **Lab #4** | **Regime-Conditioned ORB** | **Volatility + volume regime gate on Lab #3 harness** |
| Lab #5 (planned) | Signal Combination | Composite signal families through the harness |

---

## Planned Extensions

- [ ] Regime-conditioned test for all fourteen signal families from Lab #3
- [ ] Adaptive regime thresholds — recalibrated per walk-forward formation window
- [ ] Intraday regime classification using session-level volatility clustering
- [ ] Integration with ENTER Invest backtesting engine

---

## Further Reading

- [arXiv:2605.04004](https://arxiv.org/abs/2605.04004) — Mesfin (2026), primary falsification study
- [arXiv:2512.12924](https://arxiv.org/abs/2512.12924) — Garg (2025), conditional applicability finding
- [López de Prado (2018) — Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) — Chapter 17, regime detection methodology
- [Pardo (2008) — The Evaluation and Optimization of Trading Strategies](https://www.amazon.com/Evaluation-Optimization-Trading-Strategies/dp/0470128011) — walk-forward validation methodology

---

## Risk Disclosure

The experimental algorithms and implementations in this file are provided for educational and research purposes only. Past performance of any modelled strategy is not indicative of future results. All algorithmic trading carries significant financial risk, including the potential total loss of capital. Nothing here constitutes financial advice. ENTER Invest does not manage client funds based on strategies described here unless explicitly contracted to do so.

---

*Algorithmic Token is published by ENTER Invest. [algorithmictoken.substack.com](https://algorithmictoken.substack.com)* 

