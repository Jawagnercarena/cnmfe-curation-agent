"""
check_candidate_counts.py -- full CNMFe candidate count (before ACORN curation) for
every cohort-5 session in vCA1/6odorDualDiffRew, with a flag outside the expected
band.  Read-only.

Per session: N = n_candidates from candidate_features.npz (every extracted
component; the curator's auto-rejects are a subset), the review set (N minus
auto-rejected), the resolved seed parameters from agent_run.log, and two quality
proxies from the full-movie images (Cn 90th percentile, PNR 90th percentile).
Flags: LOW when N < --low (default 100), HIGH when N > --high (default 300),
PENDING when the session has no candidate_features.npz yet.  Exit code 1 when any
completed session is flagged, so the script can gate a loop.

Reference from the 12 pilots at min_corr 0.70: N = 169-266 except the odor-only
day-0 of pnb99 (81) and the weak day-1 of pnb105 (79 at 0.70, 169 at 0.60).

Usage: python check_candidate_counts.py [--low 100] [--high 300] [--only-new]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

AGENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT))
from local_config import DATA_PARENT   # noqa: E402

TASK = DATA_PARENT / "vCA1" / "6odorDualDiffRew"
ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")


def quality(sd: Path):
    try:
        Cn = np.nan_to_num(np.asarray(loadmat(str(sd / "Cn.mat"))["Cn"], dtype=float))
        pnr = np.nan_to_num(np.asarray(loadmat(str(sd / "pnr.mat"))["pnr"], dtype=float))
        b = 20
        return float(np.percentile(Cn[b:-b, b:-b], 90)), float(np.percentile(pnr[b // 2:-b // 2, b // 2:-b // 2], 90))
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=300)
    ap.add_argument("--only-new", action="store_true", help="skip the 12 reviewed pilots")
    a = ap.parse_args()
    print("session | N (full) | review set | auto-rejected | min_corr / min_pnr | Cn p90 | PNR p90 | flag")
    flagged = 0
    done = 0
    pending = []
    for sd in sorted(p for p in TASK.iterdir() if p.is_dir() and any(f"-{an}-" in p.name for an in ANIMALS)):
        reviewed = (sd / "labels.mat").exists()
        if a.only_new and reviewed:
            continue
        npz = sd / "candidate_features.npz"
        if not npz.exists():
            pending.append(sd.name)
            continue
        z = np.load(str(npz), allow_pickle=True)
        n = int(np.asarray(z["n_candidates"]).ravel()[0])
        auto = int(np.asarray(z["auto_rejected"]).ravel().size) if "auto_rejected" in z.files else 0
        log = (sd / "agent_run.log").read_text(encoding="utf-8", errors="replace") if (sd / "agent_run.log").exists() else ""
        m = re.findall(r"Final parameters: \{.*?'min_corr': ([0-9.]+), 'min_pnr': ([0-9.]+)", log)
        mc, mp = (m[-1] if m else ("?", "?"))
        cn90, pnr90 = quality(sd)
        flag = "LOW" if n < a.low else ("HIGH" if n > a.high else "ok")
        if reviewed:
            flag += " (reviewed pilot)"
        elif flag != "ok":
            flagged += 1
        done += 1
        print("%s | %4d | %4d | %3d | %s / %s | %.2f | %5.1f | %s" % (sd.name[14:], n, n - auto, auto, mc, mp, cn90, pnr90, flag))
    print("\n%d session(s) complete, %d pending, %d flagged" % (done, len(pending), flagged))
    if pending:
        print("pending: " + ", ".join(p[14:] for p in pending))
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
