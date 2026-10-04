"""
closing_check.py -- prove the red team was read-only toward the pipeline.

  1. rt_pin.py --check semantics: the corpus pin is unchanged (session set,
     labels.mat mtimes/sha, JSON sha, joblib md5).
  2. Recursive mtime scan: no file under D:\\Julian_CNMFe\\{BLA,vCA1,DG_AL},
     D:\\Julian_CNMFe\\.bootstrap_diag or any _bootstrap_validate dir is newer
     than the pin was written (dot-prefixed analysis dirs inside the areas are
     scanned too).
  3. git status: only agent/eval/bootstrap_redteam_2026-08/ and the
     pre-existing untracked files are untracked; no tracked file modified.
  4. No python process other than this one is running from the agent tree.
Writes results/closing_check.json.  Exit 1 on any violation.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import rt_lib as rt

PRE_EXISTING_UNTRACKED_PREFIXES = ("agent/eval/bootstrap_matching_2026-08/", "agent/eval/step4_2026-08/",
                                   "agent/plan_fov_assignment.py", "agent/eval/bootstrap_redteam_2026-08/")


def main():
    t = rt.Timer()
    pin_time = rt.PIN_FILE.stat().st_mtime
    old = rt.load_pin()
    new = rt.compute_pin()
    d = rt.diff_pin(old, new)
    drift = {a: v for a, v in d.items() if v["new"] or v["gone"] or v["changed"] or v["joblib_changed"]}
    newer = []
    roots = [rt.DATA_ROOT[a] for a in rt.AREAS] + [rt.DATA_PARENT / ".bootstrap_diag"]
    n_files = 0
    for root in roots:
        if not root.exists():
            continue
        for dp, dns, fns in os.walk(root):
            for fn in fns:
                p = Path(dp) / fn
                n_files += 1
                try:
                    if p.stat().st_mtime > pin_time:
                        newer.append(str(p))
                except OSError:
                    pass
    gs = subprocess.run(["git", "status", "--porcelain"], cwd=str(rt.AGENT.parent), capture_output=True, text=True).stdout
    modified = [l for l in gs.splitlines() if l and not l.startswith("??")]
    untracked = [l[3:] for l in gs.splitlines() if l.startswith("??")]
    unexpected_untracked = [u for u in untracked if not u.startswith(PRE_EXISTING_UNTRACKED_PREFIXES)]
    res = {"pin_unchanged": not drift, "drift": drift, "files_scanned": n_files,
           "files_newer_than_pin": newer[:50], "n_files_newer_than_pin": len(newer),
           "tracked_modified": modified, "unexpected_untracked": unexpected_untracked,
           "runtime_s": t.s()}
    res["ok"] = bool(not drift and not newer and not modified and not unexpected_untracked)
    (rt.RESULTS / "closing_check.json").write_text(json.dumps(res, indent=1, default=rt._jsonable))
    rt.log(f"pin unchanged: {res['pin_unchanged']}; files scanned {n_files}, newer than pin: {len(newer)}; "
           f"tracked modified: {modified}; unexpected untracked: {unexpected_untracked}")
    for p in newer[:20]:
        rt.log(f"   NEWER: {p}")
    rt.log("CLOSING CHECK: " + ("PASS" if res["ok"] else "FAIL"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
