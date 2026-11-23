"""Read-only post-run check of the 0.60 re-run of 082726-pnb105: outputs fresh, params
line, threshold 0, candidate count; Cn-band composition of the new set vs the 0.70
set; where Taylor's 24 kept (0.70) cells land in the new set (cosine >= 0.45)."""
import sys, re, time, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
S = "AVG5x-TSeries-082726-pnb105-391um-35z-000"
sd = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew" / S
OLD = sd / ".prev_mincorr070_2026-10-18"; REV = sd / ".review_mincorr070_taylor_2026-10-16"
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float); return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A
ts = lambda p: time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)) if p.exists() else "MISSING"
for f in ["neuron.mat", "review_neuron.mat", "review_report.pdf", "review_summary.txt", "candidate_features.npz", "spatial_footprints.mat", "ROIs_candidates.jpg", "Cn.mat", "pnr.mat", "Ybg_weights.mat", "agent_run.log", "review_assigned.txt"]:
    p = sd / f; print("  %-24s %s  %s" % (f, ts(p), p.stat().st_size if p.exists() else ""))
log = (sd / "agent_run.log").read_text(encoding="utf-8", errors="replace")
print("params:", re.findall(r"Final parameters: (\{.*?\})", log)[-1] if "Final parameters" in log else "NOT FOUND")
rs = (sd / "review_summary.txt").read_text(encoding="utf-8", errors="replace"); print("review_summary:", " | ".join(l.strip() for l in rs.splitlines() if l.strip())[:400])
z = np.load(str(sd / "candidate_features.npz"), allow_pickle=True); N = int(np.asarray(z["n_candidates"]).ravel()[0])
ar = np.asarray(z["auto_rejected"]).ravel() if "auto_rejected" in z else None
print("n_candidates %d; auto_rejected %s" % (N, 0 if ar is None else int(ar.sum()) if ar.dtype == bool else len(ar)))
A = foot(sd / "spatial_footprints.mat"); A0 = foot(OLD / "spatial_footprints.mat"); print("footprints new %d, old 0.70 %d" % (len(A), len(A0)))
Cn = np.nan_to_num(np.asarray(loadmat(str(sd / "Cn.mat"))["Cn"], dtype=float)); m = ndi.maximum_filter(ndi.uniform_filter(Cn, 3), 9)
def cn_at(A): return np.array([m[tuple(int(x) for x in ndi.center_of_mass(a > 0.5 * a.max()))] for a in A])
c_new, c_old = cn_at(A), cn_at(A0)
for lab, c in (("new 0.60 set", c_new), ("old 0.70 set", c_old)):
    print("%s: Cn at candidate peak >= 0.85: %d | 0.75-0.85: %d | 0.70-0.75: %d | 0.60-0.70: %d | < 0.60: %d" % (
        lab, (c >= .85).sum(), ((c >= .75) & (c < .85)).sum(), ((c >= .7) & (c < .75)).sum(), ((c >= .6) & (c < .7)).sum(), (c < .6).sum()))
L = np.asarray(loadmat(str(REV / "labels.mat"))["labels"]).ravel(); keep = np.flatnonzero(L == 1)
An = A.reshape(len(A), -1); An = An / (np.linalg.norm(An, axis=1, keepdims=True) + 1e-12)
On = A0.reshape(len(A0), -1); On = On / (np.linalg.norm(On, axis=1, keepdims=True) + 1e-12)
Smat = On @ An.T
print("old 0.70 candidates with a counterpart in the new set (cos >= 0.45): %d of %d; Taylor's kept: %d of %d (min best cosine %.2f)" % (
    (Smat.max(1) >= .45).sum(), len(A0), (Smat[keep].max(1) >= .45).sum(), len(keep), Smat[keep].max(1).min()))
print("new candidates with no 0.70 counterpart: %d (Cn at peak: >= 0.70: %d, 0.60-0.70: %d, < 0.60: %d)" % (
    (Smat.max(0) < .45).sum(), ((Smat.max(0) < .45) & (c_new >= .7)).sum(), ((Smat.max(0) < .45) & (c_new >= .6) & (c_new < .7)).sum(), ((Smat.max(0) < .45) & (c_new < .6)).sum()))
