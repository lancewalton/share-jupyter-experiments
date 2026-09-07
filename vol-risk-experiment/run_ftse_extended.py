"""Index-level regime + vol-targeting on the EXTENDED FTSE (1990->2026-08),
covering the 2022 bear that the old 2021-truncated data missed."""
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM
from mc.data import load_close
from mc.returns import log_returns
from mc.forecasters import _ewma_vol_series

EXT = "/Users/lance/Projects/FTSEData/all_extended.csv"
ANN, SCALE = 252, 100.0

def _filter(x, mu, v, A, s):
    n,k=len(x),len(mu); lp=-0.5*(np.log(2*np.pi*v)+(x[:,None]-mu)**2/v)
    pdf=np.exp(lp-lp.max(1,keepdims=True)); a=np.empty((n,k)); a[0]=s*pdf[0]; a[0]/=a[0].sum()
    for t in range(1,n): a[t]=pdf[t]*(a[t-1]@A); a[t]/=a[t].sum()
    return a
def _stats(x):
    x=np.nan_to_num(x); ar=x.mean()*ANN; av=x.std()*np.sqrt(ANN); g=np.cumsum(x)
    return ar, av, (ar/av if av>0 else 0), float((g-np.maximum.accumulate(g)).min())

s = load_close(EXT); dates = s.index
r = pd.Series(log_returns(s.to_numpy()), index=dates[1:]); r = r[np.isfinite(r)]
print(f"extended FTSE: {len(r)} returns  {r.index.min().date()} -> {r.index.max().date()}\n")

# vol-targeting (causal EWMA vol, 15% target, cap 3), full + 2022 + post-2021 OOS
sig = np.sqrt(_ewma_vol_series(r.to_numpy(), lam=0.94, burn=60))*np.sqrt(ANN)
w = np.nan_to_num(np.clip(0.15/np.where(sig>0,sig,np.nan),0,3)); vm = w*r.to_numpy()
def show(mask,label):
    idx=np.where(mask)[0]
    bh=_stats(r.to_numpy()[idx]); v=_stats(vm[idx])
    print(f"  {label:<16} B&H  Sharpe {bh[2]:+.2f} maxDD {bh[3]:+.2f}   "
          f"vol-managed Sharpe {v[2]:+.2f} maxDD {v[3]:+.2f}")
yr = r.index.year.to_numpy()
print("Vol-targeting on the FTSE INDEX:")
show(np.ones(len(r),bool), "full 1990-2026")
show(yr>=2013, "2013-2026")
show(yr==2022, "2022 bear only")

# regime: fit on first 70%, causal filter, annual % turbulent (does it flag 2022?)
split=int(0.7*len(r))
hmm=GaussianHMM(n_components=2,covariance_type="diag",n_iter=200,random_state=0).fit((r.to_numpy()[:split]*SCALE).reshape(-1,1))
turb=int(np.argmax(hmm.covars_.ravel()))
P=_filter(r.to_numpy()*SCALE,hmm.means_.ravel(),hmm.covars_.ravel(),hmm.transmat_,hmm.startprob_)[:,turb]
vol_state=np.sqrt(hmm.covars_.ravel())/SCALE*np.sqrt(ANN)
print(f"\nRegime: calm {vol_state[1-turb]:.0%} vol, turbulent {vol_state[turb]:.0%} vol")
print("  % of days in TURBULENT regime by year (recent):")
for y0 in range(2018,2027):
    m=yr==y0
    if m.sum(): print(f"    {y0}: {P[m].mean()*100:4.0f}%   (mean return {r.to_numpy()[m].mean()*252:+.1%})")
