"""Generate the experiment write-up as a self-contained HTML page.

Writes two files from one source:
  - synthesis.html      : full standalone page (matches the repo's other experiments)
  - writeup_body.html   : body-only (title + style + markup) for publishing as an Artifact
Charts are embedded as base64 data URIs so both files are fully self-contained.
"""
from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "charts"


def data_uri(name: str) -> str:
    b = (CHARTS / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


IMG = {n: data_uri(f"{n}.png") for n in
       ["ftse_structure_1y", "signals", "backtest", "backtest_span", "setup_analysis", "futures"]}

STYLE = """
:root{
  --paper:#FBFCFD; --panel:#F1F4F8; --ink:#14181F; --muted:#5A6472;
  --line:#D9DFE7; --loss:#B23A3A; --ref:#1F6F78; --accent:#2A3550;
  --loss-soft:#F3E1DF; --ref-soft:#DDECEC;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0F1319; --panel:#161C24; --ink:#E7ECF3; --muted:#97A2B2;
    --line:#28313D; --loss:#D2645C; --ref:#5FB0A8; --accent:#AEBAD2;
    --loss-soft:#2A1B1A; --ref-soft:#132422;
  }
}
:root[data-theme="dark"]{
  --paper:#0F1319; --panel:#161C24; --ink:#E7ECF3; --muted:#97A2B2;
  --line:#28313D; --loss:#D2645C; --ref:#5FB0A8; --accent:#AEBAD2;
  --loss-soft:#2A1B1A; --ref-soft:#132422;
}
*{box-sizing:border-box}
body{
  background:var(--paper); color:var(--ink); margin:0;
  font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;
  font-size:18px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:760px; margin:0 auto; padding:0 26px 96px}
.mono{font-family:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace}
.sans{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif}
h1,h2,h3{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  text-wrap:balance; line-height:1.15; letter-spacing:-.01em}
a{color:var(--ref)}

/* Masthead */
header{padding:64px 0 34px; border-bottom:2px solid var(--ink)}
.eyebrow{font-family:"IBM Plex Mono",monospace; font-size:12px;
  letter-spacing:.22em; text-transform:uppercase; color:var(--muted)}
h1{font-size:44px; font-weight:700; margin:.32em 0 .18em}
.dek{font-size:20px; color:var(--muted); margin:0; max-width:60ch}
.byline{font-family:"IBM Plex Mono",monospace; font-size:12.5px;
  color:var(--muted); margin-top:20px; letter-spacing:.02em}

/* Stat row */
.stats{display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--line); border:1px solid var(--line); margin:30px 0 0}
.stat{background:var(--paper); padding:15px 16px}
.stat .n{font-family:"IBM Plex Mono",monospace; font-size:25px; font-weight:600;
  color:var(--loss); font-variant-numeric:tabular-nums; letter-spacing:-.02em}
.stat .l{font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:var(--muted);
  text-transform:uppercase; letter-spacing:.12em; margin-top:5px; line-height:1.4}

/* Verdict panel */
.verdict{background:var(--loss-soft); border-left:3px solid var(--loss);
  padding:22px 26px; margin:40px 0; border-radius:0 3px 3px 0}
.verdict p{margin:0; font-size:20px; line-height:1.5}
.verdict .tag{font-family:"IBM Plex Mono",monospace; font-size:11px; font-weight:600;
  letter-spacing:.18em; text-transform:uppercase; color:var(--loss); display:block; margin-bottom:8px}

/* Sections */
section{padding-top:52px}
.num{font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--ref);
  letter-spacing:.1em; font-weight:600}
h2{font-size:27px; font-weight:600; margin:6px 0 14px}
h3{font-size:18px; font-weight:600; margin:26px 0 6px; color:var(--accent)}
p{margin:0 0 16px}
strong{font-weight:600}
.lead{font-size:19px}

/* Figure */
figure{margin:28px 0; padding:0}
figure img{display:block; width:100%; height:auto; border:1px solid var(--line);
  border-radius:3px; background:#fff}
figcaption{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--muted);
  margin-top:9px; line-height:1.5}

/* Table */
.tbl{width:100%; overflow-x:auto; margin:22px 0}
table{border-collapse:collapse; width:100%; font-family:"IBM Plex Mono",monospace;
  font-size:13.5px; font-variant-numeric:tabular-nums}
th,td{text-align:right; padding:8px 12px; border-bottom:1px solid var(--line); white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase;
  letter-spacing:.08em; border-bottom:1.5px solid var(--ink)}
tbody tr.hi td{background:var(--loss-soft)}
td.neg{color:var(--loss)} td.pos{color:var(--ref)}
caption{caption-side:bottom; font-family:"IBM Plex Mono",monospace; font-size:11.5px;
  color:var(--muted); text-align:left; padding-top:9px; line-height:1.5}

/* Inline emphasis chips */
.chip{font-family:"IBM Plex Mono",monospace; font-size:.86em; background:var(--panel);
  border:1px solid var(--line); border-radius:3px; padding:1px 6px; white-space:nowrap}

ul{margin:0 0 16px; padding-left:1.15em}
li{margin:0 0 8px}

footer{margin-top:64px; padding-top:24px; border-top:1px solid var(--line);
  font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--muted); line-height:1.7}
footer code{background:var(--panel); padding:1px 5px; border-radius:3px}
@media (max-width:560px){
  h1{font-size:34px} .stats{grid-template-columns:repeat(2,1fr)}
  body{font-size:17px}
}
"""

BODY = f"""<title>Trend-Line Swing Trading, Tested</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{STYLE}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Quantitative research note · FTSE daily bars</div>
  <h1>Trend-Line Swing Trading, Tested</h1>
  <p class="dek">A popular support/resistance breakout method, implemented faithfully and
  measured against a proper cost model. The idea is clean and fully mechanisable — and it
  does not work.</p>
  <div class="byline">Lance Walton · 6 September 2026 · 20 FTSE names · ~26 years · spread-bet costs</div>
  <div class="stats">
    <div class="stat"><div class="n">0 / 20</div><div class="l">names profitable</div></div>
    <div class="stat"><div class="n">0.72</div><div class="l">profit factor (gross)</div></div>
    <div class="stat"><div class="n">&minus;83%</div><div class="l">median vs +251% buy &amp; hold</div></div>
    <div class="stat"><div class="n">k · S · exit</div><div class="l">robust across all three</div></div>
  </div>
</header>

<div class="verdict">
  <span class="tag">Verdict</span>
  <p>The method, as it can be faithfully specified on daily bars, has <strong>negative expectancy
  before costs</strong>, and the result is robust to every lever it exposes — the volatility band,
  the trend-line length filter, and the exit rule. It is a rigorously falsified idea.</p>
</div>

<section>
  <div class="num">01 — THE IDEA</div>
  <h2>Straight lines under lows, over highs</h2>
  <p class="lead">Draw <strong>support</strong> lines through swing lows and <strong>resistance</strong>
  lines through swing highs, extend them forward as rays, and trade the breakouts: go long when price
  breaks the nearest resistance, exit when it breaks the opposing support (the "safety line"), trailing
  that line to lock in profit. It is the archetypal chart-trader's method — and, unusually, precise
  enough to test without discretion.</p>
  <p>The claim under test is a specific author's: that this, traded on a short timeframe, makes money.
  We test the structure and rules exactly as described, on daily bars, and let the numbers speak.</p>
</section>

<section>
  <div class="num">02 — STRUCTURE</div>
  <h2>The lines are a convex hull</h2>
  <p>The construction has a clean identity: the support lines are exactly the <strong>lower convex
  hull</strong> of the lows (anchored on the all-time low, gradients increasing); resistance is the
  <strong>upper hull</strong> of the highs. That makes the whole structure layer <strong>deterministic,
  parameter-free</strong>, and O(n) to compute — every tunable choice lives later, in the signal layer.
  We work in <span class="chip">ln(price)</span>, so a straight line is constant % growth.</p>
  <p>One data note that matters more than it should: the method anchors on the single global extreme,
  so a single bad tick rewrites everything. BP printed a low of <span class="chip">4.7p</span> on
  2019-12-13 while trading around £4.61 — one glitch that hijacked 26 years of support structure until
  it was repaired.</p>
  <figure>
    <img src="{IMG['ftse_structure_1y']}" alt="Support and resistance hull lines on six FTSE stocks, last year of daily bars">
    <figcaption>The deterministic structure on a one-year window. Green = support fan, red = resistance
    fan; solid segments are the drawn lines, dashed are their forward rays. It looks exactly like what a
    human draws by hand — which is the point.</figcaption>
  </figure>
</section>

<section>
  <div class="num">03 — SIGNALS</div>
  <h2>Frequent, and firing both ways</h2>
  <p>Closeness — for a "touch" of a line, and for a "breach" of it — is set by one pre-registered band,
  <span class="chip">k · ATR</span> (14-day Wilder ATR), evaluated in price space. An entry needs the
  action line to have at least three touches, then a breakout beyond the band. Walked forward with no
  look-ahead, the signals come thick and in both directions through sideways tape — an early sign the
  breakouts are being read from noise.</p>
  <figure>
    <img src="{IMG['signals']}" alt="Long and short entry signals marked on four FTSE stocks over the last ~300 days">
    <figcaption>Walk-forward entries over ~300 days: ▲ long, ▼ short. Twenty to thirty per name per year,
    interleaved long and short — the trend-line structure is not isolating a rare, decisive event.</figcaption>
  </figure>
</section>

<section>
  <div class="num">04 — THE BACKTEST</div>
  <h2>Every name loses; buy &amp; hold wins</h2>
  <p>The full simulator enters at the next open, trails the safety-line stop with an ATR disaster-floor,
  takes one position at a time, and prices a <strong>spread-bet wrapper</strong>: 10 bps/side spread plus
  5%/yr financing, <em>no</em> UK stamp duty. Twenty liquid, sector-varied FTSE names, ~26 years each.</p>
  <div class="tbl">
  <table>
    <thead><tr><th>Base case, k = 0.5</th><th>value</th></tr></thead>
    <tbody>
      <tr class="hi"><td>Names profitable</td><td class="neg">0 / 20</td></tr>
      <tr><td>Median strategy total</td><td class="neg">&minus;83%</td></tr>
      <tr><td>Median buy &amp; hold</td><td class="pos">+251%</td></tr>
      <tr><td>Pooled win rate</td><td>14.4%</td></tr>
      <tr><td>Mean net return / trade</td><td class="neg">&minus;0.30%</td></tr>
      <tr><td>Profit factor</td><td class="neg">0.39</td></tr>
    </tbody>
    <caption>Across ~11,900 trades. Profit factor below 1.0 means gross losses exceed gross wins.</caption>
  </table>
  </div>
  <p>The <strong>k-sweep</strong> — our pre-registered robustness check — never crosses into profit:
  profit factor sits at 0.38–0.45 for every <span class="chip">k ∈ {{0.25 … 1.5}}</span>, and mean
  return per trade stays negative throughout. Not a tuning problem.</p>
  <p>Removing <em>all</em> costs is the decisive diagnostic: mean <strong>gross</strong> return is still
  <span class="chip">&minus;0.083%</span>/trade (profit factor 0.73). The signal has no edge before a
  penny of cost is charged. And average holding was <strong>1.2 days</strong> — the "swing" method never
  actually swings, because the nearest safety line trails right under price and whips the position out.</p>
  <figure>
    <img src="{IMG['backtest']}" alt="Left: per-ticker strategy vs buy and hold, all strategy bars negative. Right: k-sweep mean return per trade, always below zero">
    <figcaption>Left: strategy (red) vs buy &amp; hold (blue), per name, 26 years — every red bar is
    negative. Right: mean net return per trade across the k-sweep never reaches the zero line.</figcaption>
  </figure>
</section>

<section>
  <div class="num">05 — TRYING TO SAVE IT</div>
  <h2>The obvious fixes make it worse</h2>
  <p>The 1.2-day whipsaw points to the fragility of signalling off the <em>last</em>, steepest, newest
  line — on both entry and exit. Two fixes were tried, pre-registered.</p>
  <h3>A minimum line length (S = 20 days)</h3>
  <p>Forcing the action and safety lines to span at least a month gives trades room to breathe — holds
  rise to ~16 days, win rate to ~30%. But the edge gets <strong>monotonically worse</strong> as the lines
  grow longer, gross and net alike: a bigger move is needed to break a longer, shallower line, after which
  reversion is only stronger.</p>
  <div class="tbl">
  <table>
    <thead><tr><th>min span S</th><th>hold</th><th>gross PF</th><th>net PF</th><th>% profitable</th></tr></thead>
    <tbody>
      <tr><td>5 d</td><td>4.6 d</td><td>0.80</td><td>0.63</td><td class="neg">0%</td></tr>
      <tr><td>10 d</td><td>8.7 d</td><td>0.77</td><td>0.62</td><td class="neg">0%</td></tr>
      <tr class="hi"><td>20 d</td><td>15.8 d</td><td>0.72</td><td>0.57</td><td class="neg">0%</td></tr>
      <tr><td>40 d</td><td>26.8 d</td><td>0.68</td><td>0.53</td><td class="neg">0%</td></tr>
      <tr><td>60 d</td><td>35.0 d</td><td>0.65</td><td>0.49</td><td class="neg">0%</td></tr>
    </tbody>
    <caption>Longer, more "established" lines → worse expectancy. No plateau, no lucky corner.</caption>
  </table>
  </div>
  <h3>The author's exact exit (safety line only, no floor)</h3>
  <p>Tracking the bare safety line with no ATR floor is closer to what the author does. It reintroduces
  the whipsaw at short spans, and the 20-day minimum tames it — but at S = 20 it is
  <strong>numerically identical</strong> to the floored version (<span class="chip">gross 0.72 / net
  0.57</span>). The exit rule doesn't matter, because the entries have no edge to protect.</p>
  <figure>
    <img src="{IMG['backtest_span']}" alt="Profit factor and per-trade edge versus min span; both decline and stay below breakeven">
    <figcaption>Profit factor (left) and per-trade edge (right) versus minimum span. Gross and net both
    fall away from breakeven as span grows; the pre-registered S = 20 sits on a uniformly losing surface.</figcaption>
  </figure>
</section>

<section>
  <div class="num">06 — MULTI-TIMEFRAME</div>
  <h2>A coarser view of the same points</h2>
  <p>Wouldn't higher timeframes change the answer? Largely no: a weekly high <em>is</em> the maximum of
  that week's daily highs — coarser timeframes add no new price information, only a sparser set of the same
  extremes. And a higher-timeframe line is just a longer-span line — precisely the axis the S-sweep
  already walked, and it got monotonically worse. Multi-timeframe confluence can only <em>select a subset</em>
  of these signals, and every subset we could define lost. It is very unlikely to flip the sign.</p>
</section>

<section>
  <div class="num">07 — VERDICT</div>
  <h2>A clean negative result</h2>
  <p class="lead">On a single daily timeframe, this method has negative expectancy before costs, robust to
  the volatility band <span class="chip">k</span>, the length filter <span class="chip">S</span>, and the
  exit rule. The fragility fixes address the <em>mechanism</em> (whipsaw); none creates an <em>edge</em>.</p>
  <p><strong>Scope, honestly.</strong> This falsifies the faithful single-timeframe replication we could
  specify. It does not indict the author's full practice — live multi-timeframe reading and discretion sit
  outside what a mechanical test can capture. But nothing in the daily evidence points to those rescuing it,
  and the one structural lever they pull (longer/coarser lines) is the one we showed hurts.</p>
  <p>It was a genuinely good idea to test — precise, mechanisable, plausible. The honest answer just turned
  out to be no.</p>
</section>

<section>
  <div class="num">08 — ADDENDUM</div>
  <h2>Does the entry geometry predict success?</h2>
  <p class="lead">A natural follow-up: is the probability of a trade winning a function of the setup —
  the <strong>gradients</strong> of the action and safety lines, their <strong>separation</strong> at
  entry, and their <strong>spans</strong>? To keep this from becoming a fishing expedition, every
  relationship was fit on the earlier half of ~11,900 trades and re-checked on the later half; the
  headline is a logistic model scored on the held-out half.</p>
  <h3>Win probability — yes, and it generalises</h3>
  <p>Three effects persist in both halves: a <strong>narrow action–safety separation</strong> (a tight
  channel), a <strong>shallow safety gradient</strong>, and <strong>low wedge convergence</strong> each
  raise the win rate. Together they score an out-of-sample <span class="chip">AUC = 0.61</span> — about
  seven standard errors above chance, so this is real, generalising information. Ranked by the model's
  score, out-of-sample win rate climbs cleanly from the worst to the best geometry:</p>
  <div class="tbl">
  <table>
    <thead><tr><th>geometry decile (OOS)</th><th>win rate</th><th>gross / trade</th><th>net / trade</th></tr></thead>
    <tbody>
      <tr><td>1 — worst</td><td>9.3%</td><td class="neg">&minus;0.089%</td><td class="neg">&minus;0.304%</td></tr>
      <tr><td>5</td><td>14.8%</td><td class="neg">&minus;0.106%</td><td class="neg">&minus;0.322%</td></tr>
      <tr><td>7</td><td>20.4%</td><td class="pos">+0.007%</td><td class="neg">&minus;0.210%</td></tr>
      <tr class="hi"><td>9</td><td>25.1%</td><td class="pos">+0.063%</td><td class="neg">&minus;0.154%</td></tr>
      <tr><td>10 — best</td><td>23.8%</td><td class="neg">&minus;0.007%</td><td class="neg">&minus;0.225%</td></tr>
    </tbody>
    <caption>Out-of-sample. Win rate is genuinely sorted by the geometry; gross return is not.</caption>
  </table>
  </div>
  <h3>Profit — no</h3>
  <p>The features sort <em>whether</em> a trade wins, not <em>how much</em>: their correlation with the
  trade return itself is ≈ 0. Tighter channels win more often but by smaller amounts, so even the
  best-geometry decile is only <strong>gross break-even</strong> (+0.06% at the very top), and after
  costs <strong>every decile is net-negative</strong>. You can predict the hit-rate; you cannot predict
  the payoff.</p>
  <figure>
    <img src="{IMG['setup_analysis']}" alt="Win rate versus each geometry feature, in-sample versus out-of-sample; separation, safety gradient and convergence trend consistently">
    <figcaption>Win rate against each feature, in-sample (blue) vs out-of-sample (orange). Separation,
    safety gradient and convergence trend consistently across both halves — but they move win
    probability, not expectancy.</figcaption>
  </figure>
  <h3>Volume behaves exactly the same</h3>
  <p>The README's other open question was volume: does a breakout on heavy volume mean more than one on
  light volume? Adding breakout-bar volume (relative to its 20-bar median) tells the identical story. It
  has ≈ 0 correlation with the trade <em>return</em>, so it does not predict move size — but it does
  modestly lift win <em>probability</em> (out-of-sample AUC rises 0.60 → <span class="chip">0.62</span>;
  win rate climbs from 15% to 20% from the lowest to the highest volume quintile). Every volume quintile
  is still gross- and net-negative:</p>
  <div class="tbl">
  <table>
    <thead><tr><th>breakout-volume quintile (OOS)</th><th>median vol×</th><th>win %</th><th>gross/trade</th><th>net/trade</th></tr></thead>
    <tbody>
      <tr><td>1 — lowest</td><td>0.68×</td><td>15.4%</td><td class="neg">&minus;0.035%</td><td class="neg">&minus;0.251%</td></tr>
      <tr><td>5 — highest</td><td>2.03×</td><td>19.7%</td><td class="neg">&minus;0.040%</td><td class="neg">&minus;0.256%</td></tr>
    </tbody>
    <caption>Both things chart-traders trust to "confirm" a breakout — a clean setup and a volume surge —
    raise the odds of a win and change nothing about the payoff.</caption>
  </table>
  </div>
  <p>This is the study's most illuminating result. It shows <em>why</em> the method is such a consistent
  near-miss: the setups it selects are essentially random on expectancy. Every lever — channel geometry
  <em>and</em> volume — re-sorts <em>whether</em> a trade wins, never <em>how much</em>, so no filter
  built from them can turn the method profitable.</p>
</section>

<section>
  <div class="num">09 — FUTURES</div>
  <h2>A different instrument, a faded premium</h2>
  <p class="lead">Single equities are hostile ground for breakouts — they mean-revert. The author,
  though, reportedly trades <strong>futures</strong>, where trends genuinely persist. Re-run on a
  diversified 22-market basket — energy, metals, ags, rates, FX, plus the S&amp;P and Nasdaq — over
  ~25 years, with futures costs (5 bps/side, no financing), the picture changes markedly.</p>
  <p>The method is <strong>far more sympathetic</strong> here: base-case gross profit factor
  <span class="chip">0.97</span> versus 0.73 on equities, with 45% of markets gross-profitable on their
  own (silver +61%, cotton, soybeans, copper). And the minimum-span lever <strong>flips sign</strong>:
  on equities longer lines hurt, but on trending futures they help monotonically — letting trades ride
  trends instead of whipsawing. Across the full sample, net turns <em>positive</em> once the lines are
  long enough (S ≥ 40).</p>
  <figure>
    <img src="{IMG['futures']}" alt="Left: profit factor rising with span on futures, crossing 1.0. Right: net profit factor by span for 2000-2012 vs 2013-2025, with the later era staying at or below 1.0">
    <figcaption>Left: on futures, longer lines lift the profit factor across break-even — the opposite of
    equities. Right: but the net edge lives almost entirely in the 2000–2012 trend era; after 2013 it
    sits on the break-even line.</figcaption>
  </figure>
  <p>That out-of-sample split is the verdict. Trained on 2000–2012 and tested on 2013–2025:</p>
  <div class="tbl">
  <table>
    <thead><tr><th>min span S</th><th>train net PF</th><th>test net PF</th><th>test net/trade</th></tr></thead>
    <tbody>
      <tr><td>40</td><td class="pos">1.09</td><td>0.99</td><td class="neg">&minus;0.020%</td></tr>
      <tr class="hi"><td>60</td><td class="pos">1.18</td><td>0.96</td><td class="neg">&minus;0.093%</td></tr>
    </tbody>
    <caption>The strong in-sample edge does not survive into the later era; test-era gross is ~1.4 SE
    from zero — indistinguishable from nothing.</caption>
  </table>
  </div>
  <p>What the method-plus-long-span actually harvested was the <strong>time-series-momentum / managed-
  futures premium</strong> — a real, documented phenomenon that was strong through the 2000s commodity
  supercycle and has been largely absent since ~2012. Both of the natural instincts here — that the
  <em>instrument</em> matters and that a <em>minimum timeframe</em> matters — were correct and visible in
  the data. They just point at a premium that has faded, not a durable inefficiency.</p>
</section>

<footer>
  METHOD · convex-hull structure in ln(price) · 14-day Wilder ATR · k·ATR touch/breach band ·
  ≥3 touches · next-open entry · trailed safety-line stop · one position at a time<br>
  COSTS · spread-bet: 10 bps/side + 5%/yr financing · no stamp duty · no CGT<br>
  DATA · 20 FTSE names + 22 futures markets · ~6,300–6,800 daily bars each · bad-tick repaired<br>
  CODE · <code>trendlines · cleaning · signals · backtest</code> · 21 TDD tests · reproducible from
  <code>backtest_run · backtest_span · backtest_pure · analyze_setup · backtest_futures*</code>
</footer>
</div>"""

# The Artifact publish wraps BODY in its own head; the standalone repo file needs a
# real <head> (title/links/style) and the markup in <body>.
head_part, body_part = BODY.split("</style>", 1)
standalone = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n"
    "<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    + head_part + "</style>\n</head>\n<body>\n" + body_part.strip() + "\n</body>\n</html>\n"
)

(HERE / "synthesis.html").write_text(standalone)
(HERE / "writeup_body.html").write_text(BODY)
print("wrote synthesis.html (%d KB) and writeup_body.html" % (len(standalone) // 1024))
