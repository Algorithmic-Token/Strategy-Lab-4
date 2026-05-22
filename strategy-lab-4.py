> **Algorithmic Token · ENTER Invest · Strategy Lab**

# Strategy Lab #4 — Does Regime Conditioning Rescue the Opening Range Breakout?

**Strategy Lab #3 falsified fourteen OHLCV signal families, including the Opening Range Breakout. It identified three directions where an edge might survive. This Lab tests Direction 1: apply a volatility and volume regime filter to the ORB and run it through the same five institutional criteria. The harness gives a verdict. We report it honestly.**

*ENTER Invest · Algorithmic Token · May 2026*

---

## A Note on These Strategy Labs

This is the fourth Strategy Lab post from Algorithmic Token. With these Strategy Labs we aim to produce a first implementation of experimental algorithmic frameworks for trading strategies, that would, conditioned on feedback and positive backtest results, be further developed into proper functional production code. The experimental algorithms generated here are based on the papers we have read and analysed, and where we have found a way for a potential trading strategy implementation. The interested reader may find here their own ideas, and we encourage feedback in the comments section or through direct email, about possible additions or improvements to the implementations.

---

## The Continuation

Strategy Lab #3 ended with three directions. We reproduce them here because the entire structure of this article depends on understanding them:

**Direction 1 — Regime-conditioned signal families.** OHLCV signals work during elevated volatility and elevated volume. A signal deliberately switched off in calm, low-volume markets is a different strategy from one that runs continuously. That strategy has not been properly tested.

**Direction 2 — Signal combination.** No single signal family passes all five criteria. A composite of two or three families, conditioned on regime agreement, has not been evaluated.

**Direction 3 — Higher-resolution data.** Several signal families require tick-level or order-book data to implement with fidelity. At tick level they may survive the harness.

We are testing Direction 1 today. The question is narrow and specific: **does adding a volatility and volume regime filter to the Opening Range Breakout produce a signal that passes the falsification harness?**

The answer is not predetermined. If the regime-conditioned ORB passes, we have found a conditional edge worth developing. If it fails, we close off Direction 1 and report where exactly it fails — which is itself a useful finding. If it partially passes — surviving some walk-forward windows but not the 75% stability threshold — we have the most instructive outcome of all: a fragile edge that tells us precisely under which conditions the ORB is tradeable.

> *"A conditional edge is still an edge. The condition is part of the strategy, not a caveat about it."*

---

## The Research Basis

This Lab does not introduce a new paper. It extends the academic framework already established in Lab #3, with one addition.

**Continuing references:**

Mesfin, M. (2026) — *Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures*
[arXiv:2605.04004](https://arxiv.org/abs/2605.04004) · q-fin.TR

Garg (2025) — *Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework*
[arXiv:2512.12924](https://arxiv.org/abs/2512.12924) · q-fin.TR

**New reference for regime detection methodology:**

López de Prado, M. (2018) — *Advances in Financial Machine Learning*
Wiley · [Amazon](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)

Chapter 17 of López de Prado covers the construction of market regimes from financial time series — the theoretical grounding for why regime filters work when they work, and why they fail when they fail. The key insight: a regime filter is only useful if it is **forward-looking in construction and backward-looking in application**. You identify the regime from data available at decision time; you never look ahead to tomorrow's volatility to decide today's trade.

---

## What the Regime Filter Does — and Does Not Do

Before the experimental algorithm, a precise statement of what we are and are not claiming for the regime filter.

**What it does:** the filter restricts trading to periods of elevated realised volatility and elevated relative volume, based on the empirical finding from Garg (2025) that OHLCV microstructure signals generate positive returns during high-information regimes and underperform in stable, low-activity markets. It does not predict which direction the market will move. It predicts nothing about the ORB signal's performance. It only identifies when the *conditions* under which OHLCV signals have historically shown any edge are present.

**What it does not do:** it does not fix a broken signal. If the ORB has no edge even in high-volatility, high-volume regimes, the regime filter will not create one. It reduces the number of trades taken, which has two effects: it reduces the noise in the signal by eliminating trades taken in unfavourable conditions, but it also reduces the trade count — which directly pressures Criterion 3 of the falsification harness (minimum 30 trades per window). A regime filter that is too restrictive will pass Criterion 2 (T-statistic) while failing Criterion 3 (trade count), producing a statistically clean but practically unusable result.

This trade-off — filter aggressiveness versus trade count — is the central tension of this Lab.

---

## Strategy Logic

The strategy has three layers, each building on the previous.

### Layer 1 — The Regime Classification

At each bar, the regime classifier asks two questions simultaneously:

```
Question 1 — Is volatility elevated?
    Compute 20-day rolling realised volatility from 5-min returns
    Compute the 60th percentile of that volatility over the past year
    Active if: current_vol > 60th_percentile_vol

Question 2 — Is volume elevated?
    Compute current bar volume relative to 60-day rolling average
    Active if: relative_volume > 1.10 (at least 10% above average)

Regime = ACTIVE if both conditions are met simultaneously
Regime = INACTIVE otherwise — no trades taken
```

The thresholds (60th percentile for volatility, 1.10x for volume) are the starting parameters from Garg (2025). They are not optimised — they are applied as published and held fixed across all walk-forward windows. Fitting these thresholds to the test data would constitute look-ahead bias and invalidate the harness.

### Layer 2 — The ORB Signal (unchanged from Lab #3)

```
Formation period: first 30 minutes of the session (6 bars at 5-min)

    ORB_high = max(High) over first 30 minutes
    ORB_low  = min(Low)  over first 30 minutes

Post-formation signal:
    If Close > ORB_high: signal = +1 (long)
    If Close < ORB_low:  signal = -1 (short)
    Otherwise:           signal =  0 (flat)

Session close: return to flat regardless of position
```

### Layer 3 — Combining Signal and Regime

```
final_signal(t) = ORB_signal(t)  IF  regime(t) == ACTIVE
final_signal(t) = 0               IF  regime(t) == INACTIVE
```

The regime filter is applied as a gate — the ORB signal is computed in full, then zeroed out in inactive regime bars. This means the regime classification must be computed using only data available at bar t — no forward-looking inputs.

---

## Experimental Algorithm Implementation

```python
# Strategy Lab #4 — Regime-Conditioned Opening Range Breakout
# Algorithmic Token · ENTER Invest
# Experimental algorithm — see risk disclosure
#
# Extends: strategy_lab_03.py (falsification harness)
# References:
#   Mesfin (2026) arXiv:2605.04004
#   Garg (2025) arXiv:2512.12924
#   López de Prado (2018) Advances in Financial Machine Learning, Ch.17

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

# Re-use harness and data functions from Strategy Lab #3
# In the repository, import directly:
# from strategy_lab_03.strategy_lab_03 import (
#     get_intraday_data,
#     run_falsification_harness,
# )


# ---------------------------------------------------------------------------
# Data acquisition (carried over from Lab #3)
# ---------------------------------------------------------------------------

def get_intraday_data(ticker: str = "MNQ=F",
                      period: str = "60d",
                      interval: str = "5m") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Regime filter — two-condition gate
# ---------------------------------------------------------------------------

def compute_regime_filter(df: pd.DataFrame,
                           vol_lookback_days: int = 20,
                           vol_percentile: float = 0.60,
                           volume_lookback_days: int = 60,
                           volume_ratio_min: float = 1.10,
                           bars_per_day: int = 78) -> pd.Series:
    """
    Classify each bar as ACTIVE or INACTIVE regime.

    ACTIVE requires both conditions simultaneously:
        1. Realised volatility above the vol_percentile threshold
        2. Relative volume above volume_ratio_min

    Thresholds are from Garg (2025), arXiv:2512.12924 — applied as
    published, not fitted to test data.

    Parameters
    ----------
    df                  : pd.DataFrame — OHLCV with DatetimeIndex
    vol_lookback_days   : int   — rolling window for vol estimation (days)
    vol_percentile      : float — vol activation threshold (default 0.60)
    volume_lookback_days: int   — rolling window for volume baseline (days)
    volume_ratio_min    : float — minimum relative volume (default 1.10)
    bars_per_day        : int   — 5-min bars per session (default 78)

    Returns
    -------
    pd.Series — boolean mask (True = ACTIVE regime)
    """
    returns = df["Close"].pct_change()

    # Annualised realised volatility from 5-min returns
    rv = (returns
          .rolling(vol_lookback_days * bars_per_day)
          .std() * np.sqrt(252 * bars_per_day))

    # Rolling percentile threshold — computed from past data only
    vol_threshold = (rv
                     .rolling(252 * bars_per_day, min_periods=bars_per_day * 5)
                     .quantile(vol_percentile))

    vol_condition = rv > vol_threshold

    # Relative volume vs rolling average
    avg_volume = df["Volume"].rolling(volume_lookback_days * bars_per_day,
                                      min_periods=bars_per_day * 5).mean()
    relative_volume  = df["Volume"] / avg_volume
    volume_condition = relative_volume > volume_ratio_min

    return vol_condition & volume_condition


# ---------------------------------------------------------------------------
# Regime sensitivity analysis
# ---------------------------------------------------------------------------

def regime_sensitivity_analysis(df: pd.DataFrame,
                                 vol_percentiles: list = [0.50, 0.60, 0.70],
                                 volume_ratios: list = [1.05, 1.10, 1.20]
                                 ) -> pd.DataFrame:
    """
    Test multiple regime threshold combinations and report trade count impact.

    Critical diagnostic: a regime filter that is too aggressive reduces
    trade count below Criterion 3's minimum of 30 per window. This function
    maps the trade-off between filter aggressiveness and trade count before
    running the full harness.

    Parameters
    ----------
    df              : pd.DataFrame — OHLCV data
    vol_percentiles : list — volatility threshold candidates
    volume_ratios   : list — relative volume threshold candidates

    Returns
    -------
    pd.DataFrame — grid of (vol_pct, vol_ratio) → active_bar_pct
    """
    rows = []
    for vp in vol_percentiles:
        for vr in volume_ratios:
            regime = compute_regime_filter(
                df,
                vol_percentile=vp,
                volume_ratio_min=vr
            )
            active_pct = regime.mean() * 100
            rows.append({
                "vol_percentile":    vp,
                "volume_ratio_min":  vr,
                "active_bars_pct":   round(active_pct, 1),
                "approx_trades_per_63d_window": round(active_pct / 100 * 63 * 3, 0),
                # rough estimate: ~3 ORB trades per active day
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ORB signal (unchanged from Lab #3)
# ---------------------------------------------------------------------------

def opening_range_breakout_signal(df: pd.DataFrame,
                                   orb_minutes: int = 30) -> pd.Series:
    """
    Opening Range Breakout signal on 5-minute bars.
    Identical implementation to Strategy Lab #3.
    """
    orb_bars    = orb_minutes // 5
    signal      = pd.Series(0, index=df.index, dtype=int)
    session_day = df.index.normalize()

    for day in session_day.unique():
        day_mask = session_day == day
        day_data = df[day_mask]
        if len(day_data) < orb_bars + 1:
            continue
        orb_high = day_data["High"].iloc[:orb_bars].max()
        orb_low  = day_data["Low"].iloc[:orb_bars].min()
        for idx, row in day_data.iloc[orb_bars:].iterrows():
            if row["Close"] > orb_high:
                signal[idx] = 1
            elif row["Close"] < orb_low:
                signal[idx] = -1

    return signal


# ---------------------------------------------------------------------------
# Falsification harness (carried over from Lab #3)
# ---------------------------------------------------------------------------

def run_falsification_harness(df: pd.DataFrame,
                               signal: pd.Series,
                               regime_filter: pd.Series,
                               round_trip_cost_points: float = 2.0,
                               point_value: float = 2.0,
                               formation_days: int = 126,
                               test_days: int = 63,
                               min_trades: int = 30,
                               min_tstat: float = 2.0,
                               stability_threshold: float = 0.75,
                               verbose: bool = True) -> dict:
    """
    Five-criterion institutional falsification harness from Strategy Lab #3.
    Full docstring in strategy_lab_03.py.
    """
    cost_per_trade = round_trip_cost_points * point_value
    prices         = df["Close"]

    filtered_signal = signal.copy()
    filtered_signal[~regime_filter] = 0

    unique_days  = pd.Series(df.index.normalize().unique())
    n_days       = len(unique_days)
    window_start = 0
    windows      = []

    while window_start + formation_days + test_days <= n_days:
        form_end = window_start + formation_days
        test_end = form_end + test_days
        windows.append({
            "form_days": unique_days.iloc[window_start:form_end],
            "test_days": unique_days.iloc[form_end:test_end],
        })
        window_start += test_days

    if len(windows) < 4:
        return {"overall_verdict": "INSUFFICIENT DATA",
                "window_results": [], "pass_rate": 0.0,
                "n_windows": len(windows)}

    window_results = []
    for w_idx, window in enumerate(windows):
        test_mask   = df.index.normalize().isin(window["test_days"])
        test_signal = filtered_signal[test_mask]
        test_price  = prices[test_mask]

        position      = test_signal.shift(1).fillna(0)
        bar_pnl       = position * test_price.diff() * point_value
        trade_changes = test_signal.diff().abs() > 0
        bar_pnl      -= trade_changes.astype(float) * (cost_per_trade / 2)

        n_trades = int(trade_changes.sum())
        c3_pass  = n_trades >= min_trades

        if n_trades >= 2:
            trade_pnls = bar_pnl[trade_changes].dropna().values
            tstat, _   = (stats.ttest_1samp(trade_pnls, 0)
                          if len(trade_pnls) >= 2 and trade_pnls.std() > 0
                          else (0.0, 1.0))
            c2_pass    = float(tstat) > min_tstat
        else:
            tstat, c2_pass = 0.0, False

        net_return = float(bar_pnl.sum())
        c4_pass    = net_return > 0
        window_pass = c2_pass and c3_pass and c4_pass

        result = {
            "window": w_idx + 1, "n_trades": n_trades,
            "t_stat": round(float(tstat), 3),
            "net_return": round(net_return, 2),
            "c2_tstat": c2_pass, "c3_trades": c3_pass,
            "c4_net_ret": c4_pass, "pass": window_pass,
        }
        window_results.append(result)

        if verbose:
            status  = "PASS" if window_pass else "FAIL"
            reasons = ([f"T={tstat:.2f}<{min_tstat}"] if not c2_pass else [])
            reasons += ([f"Trades={n_trades}<{min_trades}"] if not c3_pass else [])
            reasons += ([f"Net={net_return:+.1f}"] if not c4_pass else [])
            print(f"  Window {w_idx+1:02d} [{status}]  "
                  f"Trades={n_trades:3d} | T={tstat:5.2f} | "
                  f"Net=${net_return:+8.1f}  —  "
                  f"{' | '.join(reasons) if reasons else 'all criteria met'}")

    pass_rate = sum(r["pass"] for r in window_results) / len(window_results)
    c5_pass   = pass_rate >= stability_threshold

    if verbose:
        print()
        print(f"  Pass rate          : {pass_rate:.1%}  "
              f"(threshold: {stability_threshold:.0%})")
        print(f"  ── OVERALL VERDICT : {'PASS' if c5_pass else 'FAIL'} ──")

    return {
        "window_results":  window_results,
        "overall_verdict": "PASS" if c5_pass else "FAIL",
        "pass_rate":       pass_rate,
        "n_windows":       len(window_results),
    }


# ---------------------------------------------------------------------------
# Main — full regime-conditioned ORB evaluation
# ---------------------------------------------------------------------------

def run_regime_conditioned_orb(ticker: str = "MNQ=F",
                                period: str = "60d",
                                vol_percentile: float = 0.60,
                                volume_ratio_min: float = 1.10,
                                round_trip_cost: float = 2.0,
                                verbose: bool = True) -> dict:
    """
    Full pipeline: data → regime → ORB signal → falsification harness.

    Runs both unconditional and regime-conditioned ORB through the harness
    for direct comparison.

    Parameters
    ----------
    ticker           : str   — Yahoo Finance ticker
    period           : str   — data period (max '60d' for 5m via yfinance)
    vol_percentile   : float — volatility regime threshold
    volume_ratio_min : float — volume regime threshold
    round_trip_cost  : float — round-trip cost in index points
    verbose          : bool  — print diagnostics

    Returns
    -------
    dict with keys: unconditional, regime_conditioned, regime_stats
    """
    if verbose:
        print("=" * 60)
        print("Strategy Lab #4 — Regime-Conditioned ORB")
        print("Algorithmic Token · ENTER Invest")
        print("=" * 60)
        print()

    df     = get_intraday_data(ticker, period=period, interval="5m")
    signal = opening_range_breakout_signal(df)
    regime = compute_regime_filter(df, vol_percentile=vol_percentile,
                                   volume_ratio_min=volume_ratio_min)

    # Regime statistics
    active_pct = regime.mean() * 100
    if verbose:
        print(f"Data loaded    : {len(df)} bars, "
              f"{df.index.normalize().nunique()} trading days")
        print(f"Regime active  : {active_pct:.1f}% of bars")
        print(f"Trade reduction: ~{100 - active_pct:.1f}% of ORB trades filtered")
        print()

    # Unconditional baseline (no regime filter — all bars active)
    no_filter = pd.Series(True, index=df.index)

    if verbose:
        print("── Unconditional ORB (baseline, no regime filter) ──")
    unconditional = run_falsification_harness(
        df, signal, no_filter,
        round_trip_cost_points=round_trip_cost,
        formation_days=15, test_days=10,   # reduced for 60d demo
        min_trades=10, verbose=verbose,
    )

    if verbose:
        print()
        print("── Regime-Conditioned ORB ──")
    conditioned = run_falsification_harness(
        df, signal, regime,
        round_trip_cost_points=round_trip_cost,
        formation_days=15, test_days=10,   # reduced for 60d demo
        min_trades=10, verbose=verbose,
    )

    # Sensitivity analysis
    if verbose:
        print()
        print("── Regime Sensitivity (active bar % by threshold) ──")
        sens = regime_sensitivity_analysis(df)
        print(sens.to_string(index=False))

    return {
        "unconditional":    unconditional,
        "regime_conditioned": conditioned,
        "regime_active_pct":  active_pct,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_regime_conditioned_orb(
        ticker           = "MNQ=F",
        period           = "60d",
        vol_percentile   = 0.60,
        volume_ratio_min = 1.10,
        round_trip_cost  = 2.0,
        verbose          = True,
    )


# ---------------------------------------------------------------------------
# Risk Disclosure
# ---------------------------------------------------------------------------
# The experimental algorithms and implementations in this file are provided
# for educational and research purposes only. Past performance is not
# indicative of future results. All algorithmic trading carries significant
# financial risk, including the potential total loss of capital. Nothing
# here constitutes financial advice. ENTER Invest does not manage client
# funds based on strategies described here unless explicitly contracted.
# ---------------------------------------------------------------------------

