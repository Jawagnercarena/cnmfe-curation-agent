"""
Step 3e: the vCA1 freeze-window kit.  Built and rehearsed now; the swap itself
is DEFERRED (see the plan's 3f).

    python swap_vca1.py materialize --arm b0   copy the winning bootstrap arm
                                               from _arms/ over each bootstrap
                                               session's candidate_features_v2.npz
    python swap_vca1.py backup                 copy every live v1 npz + the
                                               deployed joblib into _v1_backup/,
                                               sha-verified, write the manifest
    python swap_vca1.py rehearse               score 3 sessions FROM THE BACKUP
                                               BYTES with the backed-up joblib and
                                               require exact equality with
                                               preswap_scores.npz -- proves the
                                               restore path reproduces today
    python swap_vca1.py preflight              every swap precondition, no writes
    python swap_vca1.py swap --freeze          os.replace the v2 files into place
    python swap_vca1.py rollback               restore the backups byte-exact

The write paths are implemented HERE against vca1_common's constants rather than
by calling swap_v2's functions with patched module globals.  swap_v2.MANIFEST is
import-bound to the BLA backup dir (swap_v2.py:44) and do_backup() writes it
(:106); BLA's rollback manifest is live.  Re-pointing globals works and
vca1_common.configure() does it, but for the handful of functions that WRITE, an
explicit local implementation removes the failure mode entirely.  Only sha256 is
reused.

`backup`, `rehearse` and `preflight` are safe outside a freeze.  `materialize`
writes only the parallel v2 files, which nothing deployed reads.  `swap` and
`rollback` require the freeze.
"""
import argparse
import json
import os
import shutil
import sys

import joblib
import numpy as np

import vca1_common as vc

MANIFEST = vc.BK / "backup_manifest.json"
PRESWAP = vc.SP / "preswap_scores.npz"
REPORT = vc.SP / "swap_report.json"
MATERIALIZED = vc.SP / "materialize_report.json"


def sha256(p):
    from swap_v2 import sha256 as _s
    return _s(p)


def bk_name(sd):
    return f"{sd.parent.name}__{sd.name}__candidate_features.npz"


def all_sessions():
    la, bs, pend, _ = vc.classify_sessions()
    return la + bs + pend


def do_materialize(arm):
    """Copy the winning bootstrap arm over the parallel v2 file. Agent/pending unaffected."""
    if arm == vc.ARM_A:
        print("arm (a) is already what backfill wrote into candidate_features_v2.npz "
              "-- nothing to materialize.")
        rep = {"arm": arm, "n_copied": 0}
    else:
        _, bs, _, _ = vc.classify_sessions()
        rep = {"arm": arm, "sessions": {}}
        for sd in bs:
            k = vc.key(vc.rel(sd))
            src = vc.ARMS / f"{k}__{arm}.npz"
            if not src.exists():
                print(f"ABORT: missing arm file {src.name}")
                return 1
            dst = sd / vc.V2
            shutil.copy2(str(src), str(dst))
            h = sha256(dst)
            if h != sha256(src):
                print(f"ABORT: byte-verify failed after copy for {vc.rel(sd)}")
                return 1
            rep["sessions"][vc.rel(sd)] = {"arm_file": src.name, "sha256": h}
        rep["n_copied"] = len(bs)
        print(f"materialized arm {arm} into {len(bs)} bootstrap sessions' v2 files")

    # record the sha of EVERY v2 file, whatever the arm: this is what swap verifies
    rep["v2_sha256"] = {vc.rel(sd): sha256(sd / vc.V2) for sd in all_sessions()}
    MATERIALIZED.write_text(json.dumps(rep, indent=1))
    print(f"wrote {MATERIALIZED.name} ({len(rep['v2_sha256'])} v2 files)")
    return 0


def do_backup():
    vc.BK.mkdir(parents=True, exist_ok=True)
    sess = all_sessions()
    entries = {}
    for sd in sess:
        src, dst = sd / vc.V1, vc.BK / bk_name(sd)
        h = sha256(src)
        if dst.exists():
            if sha256(dst) != h:
                print(f"ABORT: existing backup differs from the live v1 file: {dst.name}\n"
                      f"  The pool moved since the last backup -- investigate before "
                      f"overwriting anything.")
                return 1
        else:
            shutil.copy2(str(src), str(dst))
            if sha256(dst) != h:
                print(f"ABORT: byte-verify failed after copy: {dst.name}")
                return 1
        entries[vc.rel(sd)] = {"backup": dst.name, "sha256": h,
                               "bytes": src.stat().st_size}
    jl_bk = vc.BK / vc.JOBLIB_BK_NAME
    h_jl = sha256(vc.JOBLIB_LIVE)
    if jl_bk.exists():
        if sha256(jl_bk) != h_jl:
            print("ABORT: existing joblib backup differs from the live joblib.")
            return 1
    else:
        shutil.copy2(str(vc.JOBLIB_LIVE), str(jl_bk))
        if sha256(jl_bk) != h_jl:
            print("ABORT: joblib byte-verify failed.")
            return 1
    jl_local = vc.JOBLIB_LIVE.parent / vc.JOBLIB_BK_NAME
    if not jl_local.exists():
        shutil.copy2(str(vc.JOBLIB_LIVE), str(jl_local))
    MANIFEST.write_text(json.dumps({"sessions": entries, "joblib_sha256": h_jl,
                                    "area": vc.AREA}, indent=1))
    print(f"backup OK: {len(entries)} v1 npz + joblib -> {vc.BK}")
    print(f"joblib also kept locally as {jl_local}")
    return 0


def do_rehearse(n=3):
    """Prove the restore path: backup bytes + backed-up joblib == today's scores."""
    if not MANIFEST.exists() or not PRESWAP.exists():
        print("ABORT: need `backup` and record_preswap_vca1.py first.")
        return 1
    man = json.loads(MANIFEST.read_text())
    model = joblib.load(str(vc.BK / vc.JOBLIB_BK_NAME))
    pre = np.load(PRESWAP, allow_pickle=True)
    # widest sessions first: most rows exercised per check
    rels = sorted(man["sessions"], key=lambda r: -man["sessions"][r]["bytes"])[:n]
    ok = True
    for rel in rels:
        npz = np.load(vc.BK / man["sessions"][rel]["backup"], allow_pickle=True)
        X = npz["feature_matrix"]
        s = model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]
        same = np.array_equal(s, pre[rel + "__scores"])
        ok &= same
        print(f"  rehearse {rel[:56]:<56} N={len(X):>4} "
              f"{'EXACTLY reproduces' if same else 'DIFFERS FROM'} the preswap fixture")
    print(f"rehearsal: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _watcher_state():
    """(running, last_log_age_seconds) for watcher_vCA1."""
    import subprocess
    import time
    running = False
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
            capture_output=True, text=True, timeout=30).stdout
        running = "watcher_vCA1" in out
    except Exception:
        try:
            out = subprocess.run(["tasklist"], capture_output=True, text=True,
                                 timeout=30).stdout
            running = "python.exe" in out
        except Exception:
            running = None
    log = vc.AGENT / "logs" / "watcher_vCA1.log"
    age = (time.time() - log.stat().st_mtime) if log.exists() else None
    return running, age


def do_preflight(strict=True):
    """Every swap precondition. No writes."""
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok   ' if cond else 'FAIL '} {msg}")

    print("preflight")
    dec = vc.SP / "gate_decision.json"
    if dec.exists():
        d = json.loads(dec.read_text())
        chk(d.get("deploy_ok"), f"gate_decision.deploy_ok = {d.get('deploy_ok')} "
                                f"(arm {d.get('arm')}, T {d.get('chosen_T')})")
        chk(not d.get("production_followon_required"),
            f"no production follow-on outstanding "
            f"(required = {d.get('production_followon_required')})")
    else:
        chk(not strict, "gate_decision.json missing (gates not decided yet)")

    running, age = _watcher_state()
    chk(running is False, f"watcher_vCA1 not running (detected: {running})")
    chk(age is None or age > 120, f"watcher_vCA1.log quiet "
                                  f"({'no log' if age is None else f'{age / 60:.0f} min'})")

    red = vc.AGENT / "eval" / "bootstrap_redteam_2026-08" / "redteam_report.md"
    chk(red.exists(), f"bootstrap red-team report present ({red.name})")

    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        live = {vc.rel(sd) for sd in all_sessions()}
        chk(live == set(man["sessions"]), "session set matches the backup manifest")
        drift = [r for r, e in man["sessions"].items()
                 if sha256(vc.DATA_ROOT / r / vc.V1) != e["sha256"]]
        chk(not drift, f"no live v1 npz changed since backup ({len(drift)} drifted)")
        chk(sha256(vc.JOBLIB_LIVE) == man["joblib_sha256"],
            "live joblib unchanged since backup")
    else:
        chk(not strict, "backup manifest missing (run `backup`)")

    missing = [vc.rel(sd) for sd in all_sessions() if not (sd / vc.V2).exists()]
    chk(not missing, f"every session has a v2 sibling ({len(missing)} missing)")

    if MATERIALIZED.exists():
        m = json.loads(MATERIALIZED.read_text())
        bad = [r for r, h in m["v2_sha256"].items()
               if sha256(vc.DATA_ROOT / r / vc.V2) != h]
        chk(not bad, f"v2 files match materialize_report ({len(bad)} differ)")
    else:
        chk(not strict, "materialize_report.json missing (run `materialize`)")

    print(f"PREFLIGHT: {'PASS' if ok else 'NOT READY'}")
    return 0 if ok else 1


def do_swap(freeze):
    if not freeze:
        print("refusing: `swap` needs --freeze (watcher stopped, exchange idle).")
        return 2
    if do_preflight(strict=True) != 0:
        print("ABORT: preflight failed.")
        return 1
    man = json.loads(MANIFEST.read_text())
    report = {}
    for sd in all_sessions():
        rel = vc.rel(sd)
        h_v2 = sha256(sd / vc.V2)
        os.replace(str(sd / vc.V2), str(sd / vc.V1))
        if sha256(sd / vc.V1) != h_v2:
            print(f"ABORT mid-swap: {rel} bytes changed across replace")
            return 1
        w = int(np.load(sd / vc.V1, allow_pickle=True)["feature_matrix"].shape[1])
        if w != 35:
            print(f"ABORT mid-swap: {rel} width {w} after replace")
            return 1
        report[rel] = {"width": w, "sha256": h_v2}
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"SWAP DONE: {len(report)} sessions now 35-col ({REPORT.name} written).")
    print("Next: flip config_vCA1.FEATURE_VERSION and _VALIDATED_THRESHOLD, then retrain.")
    return 0


def do_rollback():
    man = json.loads(MANIFEST.read_text())
    n = 0
    for rel, e in sorted(man["sessions"].items()):
        sd = vc.DATA_ROOT / rel
        shutil.copy2(str(vc.BK / e["backup"]), str(sd / vc.V1))
        if sha256(sd / vc.V1) != e["sha256"]:
            print(f"ROLLBACK VERIFY FAILED: {rel}")
            return 1
        n += 1
    shutil.copy2(str(vc.BK / vc.JOBLIB_BK_NAME), str(vc.JOBLIB_LIVE))
    if sha256(vc.JOBLIB_LIVE) != man["joblib_sha256"]:
        print("ROLLBACK VERIFY FAILED: joblib")
        return 1
    print(f"ROLLBACK DONE: {n} v1 npz + joblib restored byte-exact.\n"
          f"Still to do: git revert the config_vCA1.py flip commit, then restart "
          f"watcher_vCA1.py.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["materialize", "backup", "rehearse",
                                    "preflight", "swap", "rollback"])
    ap.add_argument("--arm", choices=[vc.ARM_A, vc.ARM_B0, vc.ARM_B1])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--lenient", action="store_true",
                    help="preflight: report missing prerequisites without failing")
    args = ap.parse_args()
    vc.configure()          # also asserts no BLA path survives anywhere
    if args.cmd == "materialize":
        if not args.arm:
            print("materialize needs --arm")
            sys.exit(2)
        sys.exit(do_materialize(args.arm))
    sys.exit({"backup": do_backup, "rehearse": do_rehearse,
              "preflight": lambda: do_preflight(strict=not args.lenient),
              "swap": lambda: do_swap(args.freeze),
              "rollback": do_rollback}[args.cmd]())
