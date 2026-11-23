"""Read-only: local maxima of the 0.70-run Cn image per pilot session, by threshold,
next to the candidate count and Taylor's kept count (where returned)."""
import os, re, numpy as np
from scipy.io import loadmat
from scipy import ndimage as ndi
ROOT = "D:/Julian_CNMFe/vCA1/6odorDualDiffRew"
NEW = ".rerun_mincorr070_2026-10-06"
BD = 20
print("session | N cand (0.70) | Cn local maxima >0.70 / >0.80 / >0.85 (37x37 window, border %d) | min_pnr used" % BD)
for d in sorted(os.listdir(ROOT)):
    sd = os.path.join(ROOT, d)
    if not os.path.isdir(os.path.join(sd, NEW)): continue
    Cn = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "Cn.mat"))["Cn"], dtype=float))
    z = np.load(os.path.join(sd, NEW, "candidate_features.npz"), allow_pickle=True)
    N = int(np.asarray(z["n_candidates"]).ravel()[0])
    sm = ndi.uniform_filter(Cn, 3)
    mx = (sm == ndi.maximum_filter(sm, size=37))
    mx[:BD, :] = mx[-BD:, :] = mx[:, :BD] = mx[:, -BD:] = False
    t = open(os.path.join(sd, "agent_run.log"), encoding="utf-8", errors="replace").read()
    m1 = re.findall(r"min_corr=(\d+(?:\.\d+)?), min_pnr=(\d+(?:\.\d+)?)", t)
    mp = m1[-1][1] if m1 else "?"
    cnt = [int((mx & (sm > th)).sum()) for th in (0.70, 0.80, 0.85)]
    print("%s | %d | %d / %d / %d | %s" % (d.replace("AVG5x-TSeries-", ""), N, cnt[0], cnt[1], cnt[2], mp))
