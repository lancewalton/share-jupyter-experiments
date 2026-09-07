# Trend-channel experiment — RESULTS

**Question:** can trading a rising channel — go long near the bottom, sell near the top, repeat — beat
buy-and-hold? The motivation was that some FTSE names show long, unbroken uptrends inside a fairly parallel
rising channel.

**Method (causal, all 120 FTSE names, ~27y — no cherry-picking the good charts).** At each bar, fit an OLS
regression channel on ln(close) over a trailing L-bar window; bands = fitted line ± 2σ of residuals
(parallel by construction; gradient = trend, width = 4σ). Long-only, acting at the next open: enter when the
channel *qualifies* (annualised gradient ≥ 10%, R² ≥ 0.80) and the close is in the bottom quarter of the
channel; exit at the top quarter, on a close below the lower band (break), or when the gradient turns
negative. Costs 10 bps/side + 5%/yr financing while held. Code: `channel.py` (fitter, unit-tested in
`test_channel.py`), `channel_backtest.py`; raw output `channel_backtest_results.txt`.

## Verdict: it does NOT beat buy-and-hold — decisively

L=250, net of costs, 7,918 trades over 120 names:

| metric | strategy | buy & hold |
| --- | --- | --- |
| median total return | **−4%** | **+160%** |
| median total return (gross, no costs) | +20% | +160% |
| median Sharpe | +0.02 | +0.28 |
| % of names it beats — total return | **24%** | — |
| % of names it beats — Sharpe | **10%** | — |
| per-trade net | +0.088% (win 38%, PF 1.06) | — |
| **time in market** | **~9%** | 100% |

**Why it fails is the ~9% time-in-market.** Buying only the bottom quarter of a channel and selling the top
quarter leaves you in cash ~91% of the time, so you forgo the upward drift that made the trend attractive in
the first place. The per-trade edge is real but tiny (+0.088% net); the oscillation you can harvest is an
order of magnitude smaller than the drift you give up by being flat.

**Robust** across window length — L=120/250/500 all give median-negative strategy totals and only 20–24% of
names beating B&H, Sharpe always worse.

**The irony:** the strongest trenders — the very "long unbroken uptrends" that motivated the idea — are
where the strategy loses by the *most* (RR +512% vs B&H +2159%; HILS +104% vs +4675%; AEP +126% vs +4226%).
In an unbroken uptrend the right move is to *hold*, not to sell each channel top and miss the continuation.

**Conclusion.** Selling near the top of a rising channel is systematically the wrong side of a name that
keeps trending; buy-and-hold wins on both return and Sharpe, across the universe and every window length.
Closed.
