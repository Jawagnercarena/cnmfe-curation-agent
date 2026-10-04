"""
b1_run_pilot.py — Phase B pilot: re-run bootstrap on ~12 sessions with the
fixed matcher, routing ALL outputs to D:\\Julian_CNMFe\\.bootstrap_diag\\
(dot-prefixed so every area scanner ignores it). Session dirs are strictly
read-only; this is verified per session by a full file/size/mtime snapshot
taken before and after the run.

Serial, one MATLAB at a time. Resumable: a pilot entry whose shadow dir already
has bootstrap_match_stats.json is skipped.

Usage (valence python, watchers OFF):
    python b1_run_pilot.py --list          # show the pilot table
    python b1_run_pilot.py --dry-run       # checks only, no MATLAB
    python b1_run_pilot.py                 # run everything pending
    python b1_run_pilot.py --only bla7     # substring filter
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_DIR))

import bootstrap_preagent as bp

DATA = Path(r"D:\Julian_CNMFe")
DIAG_ROOT = DATA / ".bootstrap_diag"

PERMISSIVE = {"min_corr": 0.30, "min_pnr": 4.0}

# (area/task/session, runtag, params_override)
PILOT = [
    # -- catastrophic cluster ------------------------------------------------
    ("vCA1/CTA/AVG2x-TSeries-10232023-961-420-335um-22z-A",      "fixed", None),
    ("vCA1/CTA/AVG2x-TSeries-10232023-921-880-780um-25z-A",      "fixed", None),
    ("vCA1/CTA/AVG2x-TSeries-10232023-921-880-780um-25z-B",      "fixed", None),
    ("vCA1/AA/AVG8x-TSeries-07012022-96-172um-35z-000",          "fixed", None),
    ("BLA/3odor/AVG5x-TSeries-042525-bla7-812um-27z-000",        "fixed", None),
    ("BLA/3odor/AVG5x-TSeries-042125-bla7-778um-27z-000",        "fixed", None),
    # -- top-damage (big curated sets) ---------------------------------------
    ("vCA1/Valence/AVG2x-TSeries-12082023-962-507um-467um-23z-000B", "fixed", None),
    ("vCA1/CTA/AVG2x-TSeries-12012023-962-525um-485um-23z-000A", "fixed", None),
    # -- one of the 9 legacy pre-agent sessions ------------------------------
    ("BLA/3odor/AVG5x-TSeries-02212025-bla3-667um-26z-000",      "fixed", None),
    # -- median controls -----------------------------------------------------
    ("BLA/3odor/AVG5x-TSeries-042525-bla15-183um-45z-000",       "fixed", None),
    ("vCA1/4CS/AVG8x-TSeries-05172022-100-300um-33z-000",        "fixed", None),
    # -- sandbox parent: end-to-end validation vs cached candidates + GT -----
    ("BLA/2tones/AVG5x-TSeries-093025-bla21-313um-38z-000",      "fixed", None),
    # -- permissive-params variants (H1: does the pre-agent era need
    #    min_corr relaxation on top of the orientation fix?) ----------------
    ("vCA1/AA/AVG8x-TSeries-07012022-96-172um-35z-000",          "permissive", PERMISSIVE),
    ("BLA/3odor/AVG5x-TSeries-042125-bla7-778um-27z-000",        "permissive", PERMISSIVE),
    ("vCA1/CTA/AVG2x-TSeries-10232023-921-880-780um-25z-A",      "permissive", PERMISSIVE),
]


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def snapshot(session_dir: Path) -> dict:
    """{relpath: (size, mtime_ns)} for every file under session_dir."""
    out = {}
    for p in session_dir.rglob("*"):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(session_dir))] = (st.st_size, st.st_mtime_ns)
    return out


def run_entry(rel: str, runtag: str, override, dry: bool) -> dict:
    session_dir = DATA / Path(rel)
    area, task = Path(rel).parts[0], Path(rel).parts[1]
    shadow = DIAG_ROOT / area / task / session_dir.name / runtag
    entry = {"session": rel, "runtag": runtag, "override": override or {}}

    if not session_dir.is_dir():
        entry["status"] = "MISSING_SESSION"
        log(f"  MISSING: {session_dir}")
        return entry
    tifs = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
    mats = [t.with_suffix(".mat") for t in tifs]
    if not tifs:
        entry["status"] = "NO_TIF"
        return entry
    if not any(m.exists() for m in mats):
        # cnmfe_choose_data would CONVERT the tif into the session dir — a
        # session write. Refuse instead.
        entry["status"] = "NO_MAT_REFUSED"
        log(f"  REFUSED (no converted .mat next to tif): {session_dir.name}")
        return entry
    if (shadow / "bootstrap_match_stats.json").exists():
        entry["status"] = "ALREADY_DONE"
        log(f"  done already: {rel} [{runtag}]")
        return entry
    if dry:
        entry["status"] = "READY"
        log(f"  ready: {rel} [{runtag}]" +
            (f" override={override}" if override else ""))
        return entry

    before = snapshot(session_dir)
    t0 = time.time()
    ok = bp.bootstrap_session(task, session_dir, tifs[0],
                              out_dir=shadow, keep_candidates=True,
                              params_override=override)
    entry["runtime_min"] = round((time.time() - t0) / 60, 1)
    after = snapshot(session_dir)
    changed = {k for k in set(before) | set(after)
               if before.get(k) != after.get(k)}
    entry["session_dir_untouched"] = not changed
    entry["session_dir_changes"] = sorted(changed)
    entry["status"] = "OK" if ok else "FAILED"
    log(f"  {entry['status']} in {entry['runtime_min']} min; "
        f"session dir untouched: {entry['session_dir_untouched']}")
    if changed:
        log(f"  !! session dir changes detected: {sorted(changed)}")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="substring filter on session path")
    args = ap.parse_args()

    entries = [(r, t, o) for r, t, o in PILOT
               if args.only is None or args.only.lower() in r.lower()]
    if args.list:
        for r, t, o in entries:
            print(f"  [{t:10s}] {r}" + (f"  {o}" if o else ""))
        return

    DIAG_ROOT.mkdir(exist_ok=True)
    results = []
    log(f"Pilot: {len(entries)} entries -> {DIAG_ROOT}")
    for i, (rel, runtag, override) in enumerate(entries, 1):
        log(f"[{i}/{len(entries)}] {rel} [{runtag}]")
        results.append(run_entry(rel, runtag, override, args.dry_run))
        # incremental log so progress survives interruption
        with open(Path(__file__).parent / "b1_pilot_log.json", "w") as f:
            json.dump(results, f, indent=1)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_done = sum(1 for r in results if r["status"] == "ALREADY_DONE")
    n_fail = sum(1 for r in results if r["status"] == "FAILED")
    log(f"Pilot pass complete: {n_ok} ran ok, {n_done} already done, "
        f"{n_fail} failed, {len(results) - n_ok - n_done - n_fail} skipped.")


if __name__ == "__main__":
    main()
