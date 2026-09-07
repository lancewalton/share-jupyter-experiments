"""Two-panel futures summary chart for the write-up (numbers from the runs)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = [0, 10, 20, 40, 60]
gross_pf = [0.97, 0.96, 1.01, 1.09, 1.11]
net_pf = [0.66, 0.89, 0.96, 1.04, 1.06]

S2 = [0, 20, 40, 60]
train_net = [0.63, 0.97, 1.09, 1.18]
test_net = [0.69, 0.95, 0.99, 0.96]

GREEN, RED, INK = "#1a8a3a", "#c02020", "#33404d"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(S, gross_pf, "o-", color=INK, label="gross")
ax1.plot(S, net_pf, "s--", color=RED, label="net (5bps/side)")
ax1.axhline(1.0, color="#888", lw=0.9, ls=":")
ax1.set_xlabel("minimum line span S (days)"); ax1.set_ylabel("profit factor")
ax1.set_title("Full sample: longer lines help on futures\n(opposite of equities)", fontsize=11)
ax1.legend(fontsize=9)

ax2.plot(S2, train_net, "o-", color=GREEN, label="train  2000–2012")
ax2.plot(S2, test_net, "s--", color=RED, label="test  2013–2025")
ax2.axhline(1.0, color="#888", lw=0.9, ls=":")
ax2.set_xlabel("minimum line span S (days)"); ax2.set_ylabel("net profit factor")
ax2.set_title("Out-of-sample: the edge is the 2000s trend era\n(gone after 2013)", fontsize=11)
ax2.legend(fontsize=9)

for ax in (ax1, ax2):
    ax.tick_params(labelsize=8)
fig.tight_layout()
out = Path(__file__).resolve().parent / "charts" / "futures.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out)
