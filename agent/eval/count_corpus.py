"""
Recount the ACORN training corpus across BOTH areas using the CORRECTED
bootstrap-vs-agent classification (train_classifier._is_bootstrap_session), so
the direct-review vs bootstrap split reflects the 9 reclassified sessions.

Reports: total labeled sessions, total candidate cells, and the direct-review
(agent) vs bootstrap split — the numbers on the ACORN figure/HTML.
"""
import sys
import numpy as np
from pathlib import Path

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
import scipy.io as sio
from train_classifier import _is_bootstrap_session

ROOTS = {"BLA": Path(r"D:\Julian_CNMFe\BLA"), "vCA1": Path(r"D:\Julian_CNMFe\vCA1")}

def n_pos_from_labels(sd):
    try:
        m = sio.loadmat(sd / "labels.mat")
        for k in ("labels", "label", "y"):
            if k in m:
                return int((np.asarray(m[k]).ravel() == 1).sum())
    except Exception:
        pass
    return 0

grand = dict(sess=0, cells=0, pos=0, direct_sess=0, direct_cells=0, boot_sess=0, boot_cells=0)
for area, root in ROOTS.items():
    a = dict(sess=0, cells=0, pos=0, direct_sess=0, direct_cells=0, boot_sess=0, boot_cells=0)
    if not root.exists():
        print(f"{area}: root missing {root}"); continue
    for td in sorted(root.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            f = sd / "candidate_features.npz"
            if not f.exists() or not (sd / "labels.mat").exists():
                continue
            n = int(np.load(f, allow_pickle=True)["feature_matrix"].shape[0])
            p = n_pos_from_labels(sd)
            a["sess"] += 1; a["cells"] += n; a["pos"] += p
            if _is_bootstrap_session(sd):
                a["boot_sess"] += 1; a["boot_cells"] += n
            else:
                a["direct_sess"] += 1; a["direct_cells"] += n
    print(f"{area:5}: {a['sess']:3d} sessions | {a['cells']:6d} cells | {a['pos']:5d} positive | "
          f"direct {a['direct_sess']:3d} sess/{a['direct_cells']:6d} cells | "
          f"bootstrap {a['boot_sess']:3d} sess/{a['boot_cells']:6d} cells")
    for k in grand: grand[k] += a[k]

print("-" * 100)
print(f"TOTAL: {grand['sess']} sessions | {grand['cells']} cells | {grand['pos']} positive")
print(f"   direct-review (agent): {grand['direct_sess']} sessions / {grand['direct_cells']} cells")
print(f"   bootstrap (matched)  : {grand['boot_sess']} sessions / {grand['boot_cells']} cells")
