"""
ingest_pilots_2026-10-28.py -- ingest the 12 cohort-5 pilot returns from inbox/Taylor
by EXPLICIT path, after moving the broken Alisia returns for the same sessions out of
ingest's view.  Dry run by default; --go acts.

Why explicit paths: a whole-inbox ingest walks inbox/Alisia before inbox/Taylor.  Three
Alisia folders hold reviews of these sessions that ingest cannot use -- two reviewed
from the pre-2026-10-06 bundle (112 / 116 labels against live sets of 266 / 228) and
one whose labels.mat cannot be read at all.  The length guard rejects the first two,
but an unreadable labels.mat passes the guard, would be copied onto the session, and
Taylor's real review would then be parked as a re-review.

Server step (OPERATOR-RUN; the server is read-only for Claude): each broken return is
RENAMED into inbox/Alisia/.stale_cohort5_2026-10-28/<session>/ -- a dot-prefixed path,
which ingest_returns.iter_sessions skips.  A rename on the same share moves nothing
and deletes nothing; the folder can be deleted by hand later.  A folder is moved only
if, measured now, its labels.mat is unreadable or its length differs from the live
review set; a readable, matching return is never touched.

Local step: agent/ingest_returns.py is run once per session with the inbox-relative
path Taylor/vCA1/6odorDualDiffRew/<session>, then the landed labels.mat is checked
against candidate_features.npz.

usage:
  python ingest_pilots_2026-10-28.py            # dry run: what would move, ingest --dry-run
  python ingest_pilots_2026-10-28.py --go
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

AGENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT))
from local_config import DATA_PARENT, EXCHANGE_ROOT, PYTHON_EXE   # noqa: E402

TASK = Path("vCA1") / "6odorDualDiffRew"
REVIEWER = "Taylor"
STALE_REVIEWER = "Alisia"
STALE_HOLD = ".stale_cohort5_2026-10-28"
PILOTS = [
    "AVG5x-TSeries-082626-pnb102-479um-3z-000",
    "AVG5x-TSeries-082726-pnb102-497um-3z-000",
    "AVG5x-TSeries-082626-pnb104-404um-3z-000",
    "AVG5x-TSeries-082726-pnb104-416um-3z-000",
    "AVG5x-TSeries-082626-pnb105-397um-35z-000",
    "AVG5x-TSeries-082726-pnb105-391um-35z-000",
    "AVG5x-TSeries-082626-pnb106-205um-4z-000",
    "AVG5x-TSeries-082726-pnb106-201um-4z-000",
    "AVG5x-TSeries-082626-pnb109-398um-26z-000",
    "AVG5x-TSeries-082726-pnb109-405um-25z-000",
    "AVG5x-TSeries-082626-pnb99-399um-3z-000",
    "AVG5x-TSeries-082826-pnb99-436um-3z-000-000",
]


def labels_len(p: Path):
    """(n, kept) or None if the file is missing or unreadable."""
    try:
        L = np.asarray(sio.loadmat(str(p))["labels"]).ravel()
        return int(L.size), int((L == 1).sum())
    except Exception:
        return None


def live_n(session_dir: Path):
    try:
        z = np.load(str(session_dir / "candidate_features.npz"), allow_pickle=True)
        n = int(np.asarray(z["n_candidates"]).ravel()[0])
        auto = np.asarray(z["auto_rejected"]).ravel()
        return n - int(auto.size)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="act (default: dry run)")
    args = ap.parse_args()
    if not EXCHANGE_ROOT:
        sys.exit("ERROR: exchange root not set (CNMFE_EXCHANGE_ROOT / agent/.env).")
    inbox = Path(EXCHANGE_ROOT) / "inbox"
    mode = "GO" if args.go else "DRY RUN"
    print(f"[{mode}] inbox {inbox}\n")

    # ---- 1. broken returns of these sessions in the other reviewer's folder -------------
    print(f"== 1. {STALE_REVIEWER}'s returns for the 12 pilot sessions")
    hold = inbox / STALE_REVIEWER / STALE_HOLD
    to_move = []
    for s in PILOTS:
        src = inbox / STALE_REVIEWER / TASK / s
        if not src.is_dir():
            continue
        n_live = live_n(DATA_PARENT / TASK / s)
        lab = labels_len(src / "labels.mat")
        if lab is None:
            why = "labels.mat missing or unreadable"
        elif n_live is not None and lab[0] != n_live:
            why = f"{lab[0]} labels vs live review set {n_live} (stale bundle)"
        else:
            print(f"   keep   {s}: readable, {lab[0]} labels == live set -- NOT touched")
            continue
        to_move.append((src, hold / s, why))
        print(f"   move   {s}: {why}")
    if not to_move:
        print("   nothing to move")
    for src, dst, why in to_move:
        if dst.exists():
            sys.exit(f"ERROR: holding folder already has {dst.name}; resolve by hand.")
        if args.go:
            hold.mkdir(exist_ok=True)
            os.rename(str(src), str(dst))
            print(f"   MOVED  {src.name} -> {STALE_REVIEWER}/{STALE_HOLD}/")
        else:
            print(f"   would rename {src}\n          -> {dst}")
    print()

    # ---- 2. explicit-path ingest of the 12 Taylor returns --------------------------------
    print(f"== 2. ingest 12 returns from inbox/{REVIEWER} by explicit path")
    results = []
    for s in PILOTS:
        rel = str(Path(REVIEWER) / TASK / s)
        src = inbox / REVIEWER / TASK / s
        if not (src / "labels.mat").exists():
            print(f"   {s}: NO RETURN in inbox/{REVIEWER} -- skipped")
            results.append((s, "no return", None, live_n(DATA_PARENT / TASK / s)))
            continue
        cmd = [PYTHON_EXE, str(AGENT / "ingest_returns.py"), rel] + ([] if args.go else ["--dry-run"])
        print(f"   {s}: running ingest_returns.py {'' if args.go else '--dry-run '}...")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(AGENT))
        tail = [l for l in r.stdout.splitlines() if l.strip()][-3:]
        for l in tail:
            print(f"      {l[:150]}")
        if r.returncode != 0:
            print(f"      !! exit code {r.returncode}; stderr: {r.stderr.strip()[-300:]}")
        landed = labels_len(DATA_PARENT / TASK / s / "labels.mat") if args.go else labels_len(src / "labels.mat")
        results.append((s, "ok" if r.returncode == 0 else f"exit {r.returncode}", landed, live_n(DATA_PARENT / TASK / s)))
    print()

    # ---- 3. summary ---------------------------------------------------------------------
    where = "local labels.mat" if args.go else "inbox labels.mat (dry run)"
    print(f"== 3. summary ({where} vs live review set)")
    bad = 0
    for s, status, lab, n in results:
        if lab is None:
            flag = "NO LABELS"
        elif n is None:
            flag = "live set unreadable"
        elif lab[0] != n:
            flag = f"LENGTH MISMATCH {lab[0]} vs {n}"
        else:
            flag = "ok"
        bad += flag != "ok"
        print(f"   {s[14:]:<34} {status:<8} labels {lab[0] if lab else '-':>4} kept {lab[1] if lab else '-':>3}  live {n if n is not None else '-':>4}  {flag}")
    print(f"\n{len(results) - bad} of {len(results)} sessions consistent; {bad} flagged.")
    if not args.go:
        print("Dry run: nothing was moved or written.  Re-run with --go to act.")


if __name__ == "__main__":
    main()
