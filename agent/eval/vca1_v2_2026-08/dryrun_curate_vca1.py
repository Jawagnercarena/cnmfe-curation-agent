"""
Step 3e: end-to-end curation dry-run at 35 columns, on a COPY.

Proves the production path works for vCA1 before any deploy: pass 1 scores the 13
base columns with the companion first-pass model, v2b + ranks are computed, pass 2
scores all 35 with the deployed model, and the review package is written.

Runs as a STANDALONE process and patches config_vCA1 BEFORE importing curator --
curator binds MODEL_DIR at import (curator.py:29), so patching afterwards is
inert.  After importing it asserts the module actually resolved to the rehearsal
dir, and that the session it is about to curate lives under .feature_expansion.
Every pending vCA1 session is out with a reviewer right now, so the live dirs are
never touched.

  python dryrun_curate_vca1.py            picks a pending session automatically
  python dryrun_curate_vca1.py --session 3odor/AVG5x-...
"""
import argparse
import shutil
import sys

import numpy as np

import vca1_common as vc

COPY_FILES = ["A.txt", "C_raw.txt", "C.txt", "S.txt", "Cn.mat", "pnr.mat",
              "Ybg_mean.mat", "spatial_footprints.mat", "neuron.mat", "Coor.mat",
              "candidate_features.npz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    args = ap.parse_args()
    vc.configure()

    _, _, pending, _ = vc.classify_sessions()
    if args.session:
        sd = vc.DATA_ROOT / args.session
    else:
        sd = min(pending, key=lambda p: int(
            np.load(p / vc.V1, allow_pickle=True)["n_candidates"][0]))
    rel = vc.rel(sd)
    print(f"dry-run curation on a COPY of {rel}")

    dst = vc.REHEARSAL / vc.key(rel)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in COPY_FILES:
        if (sd / f).exists():
            shutil.copy2(str(sd / f), str(dst / f))
    live13 = np.load(sd / vc.V1, allow_pickle=True)["feature_matrix"]
    print(f"  copied {len(list(dst.iterdir()))} files; live npz is "
          f"{live13.shape[0]}x{live13.shape[1]}")

    # patch BEFORE importing curator -- it binds MODEL_DIR at import time
    import config_vCA1
    config_vCA1.FEATURE_VERSION = 2
    config_vCA1.MODEL_DIR = vc.REHEARSAL / "model"
    import curator

    assert curator.MODEL_DIR == vc.REHEARSAL / "model", \
        f"curator bound MODEL_DIR = {curator.MODEL_DIR}, not the rehearsal dir"
    assert str(vc.REHEARSAL) in str(dst), "refusing: session copy is not under the rehearsal dir"
    assert getattr(curator.config, "FEATURE_VERSION", 1) == 2, "FEATURE_VERSION patch did not take"
    print(f"  curator.MODEL_DIR -> {curator.MODEL_DIR}  (production model untouched)")

    log_lines = []

    def log(m):
        log_lines.append(str(m))
        print(f"    {m}")

    curator.prepare_review_package(dst.name, dst, log)

    out = np.load(dst / vc.V1, allow_pickle=True)
    X = out["feature_matrix"]
    print(f"\n  RESULT: npz written {X.shape[0]}x{X.shape[1]}, "
          f"{len(out['auto_rejected'])} auto-rejected")
    ok = True
    if X.shape[1] != 35:
        print(f"  FAIL  width {X.shape[1]} != 35"); ok = False
    if not np.array_equal(X[:, :13], live13):
        d = np.abs(X[:, :13] - live13).max()
        print(f"  FAIL  first 13 columns differ from the live npz (max {d:.3e})"); ok = False
    else:
        print("  ok    first 13 columns bit-identical to the live 13-col npz "
              "(re-extraction is deterministic)")
    if not (X[:, 34] == 1).all():
        print("  FAIL  v2_present is not 1 on every row"); ok = False
    else:
        print("  ok    v2_present = 1 on all rows (full candidate set has real v2b)")
    if any("pass 1" in l for l in log_lines):
        print("  ok    two-pass scoring ran (pass 1 via the companion first-pass model)")
    else:
        print("  FAIL  no pass-1 line in the curator log"); ok = False
    for f in ("review_report.pdf", "review_neuron.mat", "review_summary.txt"):
        if (dst / f).exists():
            print(f"  ok    {f} produced")
        else:
            print(f"  FAIL  {f} missing"); ok = False
    print(f"\nDRY RUN: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
