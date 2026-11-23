"""Read-only: are Taylor's labels index-aligned with the candidate set?
For each session ingested 2026-11-20: every final (kept) footprint in the
returned spatial_footprints.mat should sit on a candidate labelled KEEP in
.pre_ingest/spatial_footprints.mat (all N candidates, npz order)."""
import sys, re
from pathlib import Path
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import local_config as lc
import review_prep as rp

DATA = Path(lc.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
PAT = re.compile(r"-09\d{4}-pnb(99|102|104|105|106|109)-")


def masks(A):
    A = A.reshape(len(A), -1)
    mx = A.max(1)
    return (A > 0.2 * mx[:, None]).astype(np.float32)


h = "%-34s %4s %4s %4s %5s | final footprints best-matching a: %5s %5s %5s %6s | kept cands w/o final (IoU<.5): %s"
print(h % ("session", "N", "keep", "nfin", "K_mat", "KEEP", "DEL", "AUTO", "none", ""))
for sd in sorted(p for p in DATA.iterdir() if p.is_dir() and PAT.search(p.name)):
    if not (sd / "labels.mat").exists():
        continue
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    n = len(npz["feature_matrix"])
    auto = set(npz["auto_rejected"].ravel().astype(int).tolist())
    lab = sio.loadmat(str(sd / "labels.mat"))["labels"].ravel().astype(int)
    rows = np.array([i for i in range(n) if i not in auto])
    full = np.full(n, -1); full[rows] = lab
    cand = sio.loadmat(str(sd / ".pre_ingest" / "spatial_footprints.mat"))["spatial_footprints"]
    fin = sio.loadmat(str(sd / "spatial_footprints.mat"))["spatial_footprints"]
    assert len(cand) == n, (sd.name, len(cand), n)
    Mc, Mf = masks(cand), masks(fin)
    inter = Mf @ Mc.T
    J = inter / np.maximum(Mf.sum(1)[:, None] + Mc.sum(1)[None, :] - inter, 1)
    bi = J.argmax(1); bj = J.max(1)
    cls = np.where(bj < 0.5, 9, full[bi])
    kept_idx = np.where(full == 1)[0]
    kept_cov = J[:, kept_idx].max(0) if len(fin) else np.zeros(len(kept_idx))
    meta = rp.read_kv(sd / "review_meta.txt")
    print(h % (sd.name.replace("AVG5x-TSeries-", "")[:34], n, int(lab.sum()), meta.get("n_final", "?"), len(fin),
               int((cls == 1).sum()), int((cls == 0).sum()), int((cls == -1).sum()), int((cls == 9).sum()),
               "%d (merges logged %s)" % (int((kept_cov < 0.5).sum()), meta.get("merges_logged", "?"))))
