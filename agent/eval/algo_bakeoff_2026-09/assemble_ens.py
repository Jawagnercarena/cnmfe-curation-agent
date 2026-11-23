"""
assemble_ens.py -- average the per-member OOF scores of separate train_cnn2.py
runs into one ensemble OOF, so a single-model run can serve as member 0 of an
ensemble and only the extra members need training.

  python assemble_ens.py --base fixed12+nbtrace --extra fixed12+nbtrace+ens3_m1-2 --out fixed12+nbtrace+ens3

Reads results/r2/<base>_<AREA>_<grouping>.npz (oof_members) and
results/r2/<extra>_<AREA>_<grouping>.npz for every cell present, checks rows
and seeds match, writes results/r2/<out>_<AREA>_<grouping>.npz with the same
keys (oof = probability mean over all members) and a small json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402

rt = B.rt
R2 = HERE / os.environ.get("BAKEOFF_RESULTS", "results") / "r2"
CELLS = [("BLA", "session"), ("BLA", "animal"), ("vCA1", "session"), ("vCA1", "animal")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True, help="comma-separated extra-member run names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    for area, grouping in CELLS:
        bp = R2 / f"{a.base}_{area}_{grouping}.npz"
        if not bp.exists():
            print(f"{area}/{grouping}: base missing, skipped")
            continue
        b = np.load(bp, allow_pickle=True)
        parts = [b["oof_members"] if "oof_members" in b else b["oof"][:, None, :]]
        members = list(b["members"]) if "members" in b else [0]
        seeds = [int(s) for s in b["seeds"]]
        ok = True
        for name in [x.strip() for x in a.extra.split(",") if x.strip()]:
            ep = R2 / f"{name}_{area}_{grouping}.npz"
            if not ep.exists():
                print(f"{area}/{grouping}: {ep.name} missing, skipped")
                ok = False
                break
            e = np.load(ep, allow_pickle=True)
            if [int(s) for s in e["seeds"]] != seeds or not np.array_equal(e["y"], b["y"]):
                raise SystemExit(f"{ep.name}: seeds or rows differ from {bp.name}")
            parts.append(e["oof_members"])
            members += list(e["members"])
        if not ok:
            continue
        allm = np.concatenate(parts, axis=1)                  # (seeds, members, rows)
        oof = np.nanmean(allm, axis=1)
        y, rev = b["y"].astype(int), b["reviewed"].astype(bool)
        aucs = [rt.auc(y[~np.isnan(o)], o[~np.isnan(o)], rev[~np.isnan(o)]) for o in oof]
        np.savez_compressed(R2 / f"{a.out}_{area}_{grouping}.npz", oof=oof, oof_members=allm, members=np.array(members),
                            y=b["y"], reviewed=b["reviewed"], has_raw=b["has_raw"], groups=b["groups"],
                            groups_cv=b["groups_cv"], names=b["names"], seeds=b["seeds"])
        (R2 / f"{a.out}_{area}_{grouping}.json").write_text(json.dumps({
            "assembled_from": [a.base] + a.extra.split(","), "members": [int(m) for m in members], "seeds": seeds,
            "reviewed_auc_per_seed_rows_with_raw": aucs, "written_at": dt.datetime.now().isoformat(timespec="seconds")},
            indent=1))
        print(f"{area}/{grouping}: {allm.shape[1]} members {members} -> {a.out}: reviewed AUC (rows with raw) "
              f"mean {np.mean(aucs):.4f} per seed {[round(x, 4) for x in aucs]}")


if __name__ == "__main__":
    main()
