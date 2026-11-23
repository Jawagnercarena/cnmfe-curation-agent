"""READ-ONLY.  For each of Taylor's returns: (1) are the bundle inputs she worked
from byte-identical in size/content to the local 0.70 run and the outbox bundle;
(2) do the returned kept footprints line up with the labels==1 candidates of the
local 0.70 candidate set (cosine similarity, C-order pixels)."""
import sys, hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent")
import local_config
EX = Path(local_config.EXCHANGE_ROOT); DP = Path(local_config.DATA_PARENT)
TASK = "vCA1/6odorDualDiffRew"; LOCAL = DP / TASK
INB = EX / "inbox" / "Taylor" / TASK; OUTB = EX / "outbox" / "Taylor" / TASK
NEW = ".rerun_mincorr070_2026-10-06"
INPUTS = ["review_summary.txt", "pnr.mat", "Cn.mat", "Ybg_weights.mat", "review_report.pdf", "review_neuron.mat"]
def md5(p, limit=None):
    h = hashlib.md5()
    with open(p, "rb") as f:
        n = 0
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b); n += len(b)
            if limit and n >= limit: break
    return h.hexdigest()[:10]
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], -1)
print("outbox bundle file names:", sorted(q.name for q in next(OUTB.iterdir()).iterdir()))
for r in sorted(p for p in INB.iterdir() if p.is_dir()):
    s = r.name
    if not (sd_ok := (LOCAL / s / NEW).exists()): continue
    if not (r / "labels.mat").exists(): continue
    sd = LOCAL / s
    print("\n==", s)
    for f in INPUTS:
        a, b, c = r / f, OUTB / s / f, sd / f
        sz = lambda p: p.stat().st_size if p.exists() else None
        same_size = sz(a) == sz(b) == sz(c)
        # Cn.mat in the session root was not re-exported by every path; compare content on small files
        if f in ("review_summary.txt", "pnr.mat") and a.exists() and c.exists():
            same = md5(a) == md5(c)
            print("   %-20s inbox %s | outbox %s | local %s | same size: %s | inbox==local md5: %s" % (f, sz(a), sz(b), sz(c), same_size, same))
        elif f == "review_neuron.mat" and a.exists() and c.exists():
            print("   %-20s inbox %s | outbox %s | local %s | same size: %s | first 8MB md5 inbox==local: %s" % (f, sz(a), sz(b), sz(c), same_size, md5(a, 8 << 20) == md5(c, 8 << 20)))
        else:
            print("   %-20s inbox %s | outbox %s | local %s | same size: %s" % (f, sz(a), sz(b), sz(c), same_size))
    L = np.asarray(loadmat(str(r / "labels.mat"))["labels"]).ravel()
    A_new = foot(sd / NEW / "spatial_footprints.mat")
    A_ret = foot(r / "spatial_footprints.mat")
    keep = np.flatnonzero(L == 1)
    print("   labels n=%d kept=%d | local 0.70 candidates=%d | returned footprints=%d" % (L.size, keep.size, A_new.shape[0], A_ret.shape[0]))
    if L.size != A_new.shape[0]:
        print("   !! LENGTH MISMATCH"); continue
    K = A_new[keep]
    Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
    Rn = A_ret / (np.linalg.norm(A_ret, axis=1, keepdims=True) + 1e-12)
    S = Rn @ Kn.T                      # returned x labelled-kept
    best = S.max(axis=1)
    print("   returned kept footprint -> best cosine vs labelled-kept 0.70 candidate: median %.3f, min %.3f, n<0.45: %d" % (
        np.median(best), best.min(), int((best < 0.45).sum())))
    # control: against the OLD est85 set (should be worse if she reviewed the new bundle)
    old = sd / ".prev_est85_2026-10-05" / "spatial_footprints.mat"
    if old.exists():
        A_old = foot(old); On = A_old / (np.linalg.norm(A_old, axis=1, keepdims=True) + 1e-12)
        bo = (Rn @ On.T).max(axis=1)
        print("   control vs OLD est85 set (%d cand): median %.3f, n<0.45: %d" % (A_old.shape[0], np.median(bo), int((bo < 0.45).sum())))
