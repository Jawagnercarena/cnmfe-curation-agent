"""
Step 6/7 freeze-window tool: backup, rehearse, swap, rollback.

    python swap_v2.py backup    Step 6.4: copy every v1 candidate_features.npz
                                (sessions that have a v2 sibling) into
                                D:\\...\\.feature_expansion\\_v1_backup\\ and the
                                deployed joblib alongside; byte-verify; write
                                backup_manifest.json.
    python swap_v2.py rehearse  Step 6.5: score 3 sessions (incl. the smoke
                                session) FROM THE BACKUP BYTES with the
                                backed-up joblib and require exact equality
                                with preswap_scores.npz — proves the restore
                                path reproduces pre-swap behavior.
    python swap_v2.py swap      Step 7.2: preconditions (backups byte-current,
                                no labeled/pending session missing a v2
                                sibling), then os.replace each
                                candidate_features_v2.npz over the live file.
                                Writes swap_report.json.
    python swap_v2.py rollback  Restore every backup over the live files and
                                the old joblib over classifier.joblib.
                                (Then: git revert the config flip; restart
                                the watcher.)

BLA only.  Run ONLY inside the freeze (BLA watcher stopped, exchange idle),
except `backup` and `rehearse` which are read-only on live state.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

SP = Path(__file__).parent
AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
DATA_ROOT = Path(r"D:\Julian_CNMFe\BLA")
BK = Path(r"D:\Julian_CNMFe\BLA\.feature_expansion\_v1_backup")
V2 = "candidate_features_v2.npz"
V1 = "candidate_features.npz"
JOBLIB_LIVE = AGENT / "model" / "BLA" / "classifier.joblib"
JOBLIB_BK_NAME = "classifier_v1_2026-08.joblib"
MANIFEST = BK / "backup_manifest.json"
SMOKE_REL = "2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sessions_with_v2():
    out = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if sd.is_dir() and (sd / V2).exists():
                out.append(sd)
    return out


def bk_name(sd: Path) -> str:
    return f"{sd.parent.name}__{sd.name}__candidate_features.npz"


def do_backup():
    BK.mkdir(exist_ok=True)
    sess = sessions_with_v2()
    entries = {}
    for sd in sess:
        src, dst = sd / V1, BK / bk_name(sd)
        h_src = sha256(src)
        if dst.exists():
            if sha256(dst) != h_src:
                print(f"ABORT: existing backup differs from live v1: {dst.name}\n"
                      f"  The pool moved since the last backup — investigate "
                      f"before overwriting anything.")
                return 1
        else:
            shutil.copy2(str(src), str(dst))
            if sha256(dst) != h_src:
                print(f"ABORT: byte-verify failed after copy: {dst.name}")
                return 1
        entries[f"{sd.parent.name}/{sd.name}"] = {
            "backup": dst.name, "sha256": h_src,
            "bytes": src.stat().st_size}
    jl_bk = BK / JOBLIB_BK_NAME
    h_jl = sha256(JOBLIB_LIVE)
    if jl_bk.exists():
        if sha256(jl_bk) != h_jl:
            print("ABORT: existing joblib backup differs from the live joblib.")
            return 1
    else:
        shutil.copy2(str(JOBLIB_LIVE), str(jl_bk))
        if sha256(jl_bk) != h_jl:
            print("ABORT: joblib byte-verify failed.")
            return 1
    jl_local = JOBLIB_LIVE.parent / JOBLIB_BK_NAME
    if not jl_local.exists():
        shutil.copy2(str(JOBLIB_LIVE), str(jl_local))
    MANIFEST.write_text(json.dumps(
        {"sessions": entries, "joblib_sha256": h_jl}, indent=1))
    print(f"backup OK: {len(entries)} v1 npz + joblib -> {BK}")
    print(f"joblib also kept locally as {jl_local}")
    return 0


def do_rehearse():
    man = json.loads(MANIFEST.read_text())
    model = joblib.load(BK / JOBLIB_BK_NAME)
    pre = np.load(SP / "preswap_scores.npz", allow_pickle=True)
    rels = [SMOKE_REL] + [r for r in sorted(man["sessions"]) if r != SMOKE_REL][:2]
    ok = True
    for rel in rels:
        npz = np.load(BK / man["sessions"][rel]["backup"], allow_pickle=True)
        X = npz["feature_matrix"]
        s = model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]
        same = np.array_equal(s, pre[rel + "__scores"])
        ok &= same
        print(f"  rehearse {rel}: scores from backup bytes "
              f"{'EXACTLY reproduce' if same else 'DIFFER from'} preswap fixture")
    print(f"rehearsal: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def do_swap():
    if not MANIFEST.exists():
        print("ABORT: no backup manifest — run `backup` first.")
        return 1
    man = json.loads(MANIFEST.read_text())
    sess = sessions_with_v2()
    by_rel = {f"{sd.parent.name}/{sd.name}": sd for sd in sess}

    # completeness: nothing trainable/curatable may stay 13-col
    missing = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir() or not (sd / V1).exists():
                continue
            rel = f"{td.name}/{sd.name}"
            if rel not in by_rel:
                missing.append(rel)
    if missing:
        print(f"ABORT: {len(missing)} session(s) have a v1 npz but NO v2 "
              f"sibling (would stay 13-col after the swap):")
        for r in missing:
            print(f"  {r}")
        return 1
    if set(by_rel) != set(man["sessions"]):
        print("ABORT: session set differs from the backup manifest — "
              "re-run `backup`.")
        return 1
    if sha256(JOBLIB_LIVE) != man["joblib_sha256"]:
        print("ABORT: live joblib changed since backup (a retrain ran?).")
        return 1
    for rel, sd in sorted(by_rel.items()):
        if sha256(sd / V1) != man["sessions"][rel]["sha256"]:
            print(f"ABORT: live v1 npz changed since backup: {rel} — "
                  f"the pool moved; redo backup (and re-check the freeze).")
            return 1

    import os
    report = {}
    for rel, sd in sorted(by_rel.items()):
        h_v2 = sha256(sd / V2)
        os.replace(str(sd / V2), str(sd / V1))
        if sha256(sd / V1) != h_v2:
            print(f"ABORT mid-swap: {rel} bytes changed across replace?!")
            return 1
        npz = np.load(sd / V1, allow_pickle=True)
        w = int(npz["feature_matrix"].shape[1])
        report[rel] = {"width": w, "sha256": h_v2}
        if w != 35:
            print(f"ABORT mid-swap: {rel} width {w} after replace?!")
            return 1
    (SP / "swap_report.json").write_text(json.dumps(report, indent=1))
    print(f"SWAP DONE: {len(report)} sessions now 35-col "
          f"(swap_report.json written). Next: retrain with "
          f"--prospective-only --model xgboost --threshold <T>.")
    return 0


def do_rollback():
    man = json.loads(MANIFEST.read_text())
    n = 0
    for rel, e in sorted(man["sessions"].items()):
        sd = DATA_ROOT / rel
        shutil.copy2(str(BK / e["backup"]), str(sd / V1))
        if sha256(sd / V1) != e["sha256"]:
            print(f"ROLLBACK VERIFY FAILED: {rel}")
            return 1
        n += 1
    shutil.copy2(str(BK / JOBLIB_BK_NAME), str(JOBLIB_LIVE))
    if sha256(JOBLIB_LIVE) != man["joblib_sha256"]:
        print("ROLLBACK VERIFY FAILED: joblib")
        return 1
    print(f"ROLLBACK DONE: {n} v1 npz + joblib restored byte-exact.\n"
          f"Still to do: git revert the config.py FEATURE_VERSION commit, "
          f"then restart the BLA watcher.")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"backup": do_backup, "rehearse": do_rehearse,
          "swap": do_swap, "rollback": do_rollback}.get(cmd)
    if fn is None:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn())
