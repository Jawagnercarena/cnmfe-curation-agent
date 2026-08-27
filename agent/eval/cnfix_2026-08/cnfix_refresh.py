"""
cnfix_refresh.py -- refresh the retro-labeled BLA sessions after the
cn_correlation orientation fix (red team 2026-08-26, attack #4).

The retro path (train_classifier._retro_label_session) reshaped MATLAB footprint
columns row-major, so column 12 (cn_correlation) of every retro-labeled session
was computed on TRANSPOSED footprints.  Six BLA sessions are affected (705 rows /
217 reals, all in the CV pool).  This kit rebuilds those sessions' live
candidate_features.npz with the corrected column and the 13 recomputed rank
columns, everything else byte-identical.

    python cnfix_refresh.py identify   which sessions carry the transposed column
                                       (numeric: stored col 12 == order-C recompute)
    python cnfix_refresh.py build      write corrected files to _cnfix/ (parallel,
                                       nothing deployed reads them) + cnfix_report.json
    python cnfix_refresh.py backup     copy the affected live files + the live joblib
                                       to _cnfix_backup/ (NOT the retired _v1_backup)
    python cnfix_refresh.py gate       8-seed paired OOF, live vs corrected, same folds
                                       (independent evaluator from the red team) ->
                                       cnfix_gate.json incl. the rule-chosen T
    python cnfix_refresh.py preflight  every swap precondition, no writes
    python cnfix_refresh.py swap --freeze   os.replace the corrected files into place
    python cnfix_refresh.py verify     joblib contract + corpus identity after retrain
    python cnfix_refresh.py rollback   restore the backups byte-exact

BLA only.  `swap` and `rollback` need the BLA watcher stopped.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio

SP = Path(__file__).resolve().parent
AGENT = SP.parents[1]
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(AGENT / "eval" / "bootstrap_redteam_2026-08"))
import features as F                  # noqa: E402  (the FIXED helper lives here)

DATA_ROOT = Path(r"D:\Julian_CNMFe\BLA")
EXT = DATA_ROOT / ".feature_expansion"
BUILT = EXT / "_cnfix"
BK = EXT / "_cnfix_backup"
V1 = "candidate_features.npz"
JOBLIB_LIVE = AGENT / "model" / "BLA" / "classifier.joblib"
JOBLIB_BK_NAME = "classifier_pre_cnfix_2026-08-26.joblib"
REPORT = SP / "cnfix_report.json"
MANIFEST = BK / "backup_manifest.json"
SWAP_REPORT = SP / "cnfix_swap_report.json"
GATE = SP / "cnfix_gate.json"
CN_COL = 12
SMOKE_REL = "2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"
SMOKE_IDX = [21, 24]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(sd):
    return f"{sd.parent.name}/{sd.name}"


def key(r):
    return r.replace("/", "__")


def agent_sessions():
    out = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if (sd.is_dir() and (sd / V1).exists() and (sd / "labels.mat").exists()
                    and not (sd / "bootstrap_match_stats.json").exists()):
                out.append(sd)
    return out


def load_ext(r):
    p = EXT / f"{key(r)}.mat"
    m = sio.loadmat(str(p))
    d1, d2 = int(np.asarray(m["d1"]).flat[0]), int(np.asarray(m["d2"]).flat[0])
    A = m["A"]
    A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    return np.asarray(A, float), np.asarray(m["Cn"], float), d1, d2


def reviewed_mask(npz):
    n = int(npz["n_candidates"][0])
    rv = np.ones(n, bool)
    rv[npz["auto_rejected"].astype(int)] = False
    return rv


def do_identify(verbose=True):
    retro = []
    for sd in agent_sessions():
        r = rel(sd)
        npz = np.load(sd / V1, allow_pickle=True)
        A, Cn_ext, d1, d2 = load_ext(r)
        rv = reviewed_mask(npz)
        stored = npz["feature_matrix"][rv, CN_COL]
        imgsF = F.fcols_to_images(A, d1, d2)
        imgsC = A.T.reshape(A.shape[1], d1, d2)              # the old (wrong) reshape
        Cn = sio.loadmat(str(sd / "Cn.mat"))["Cn"]
        cF = np.array([F.cn_features(imgsF[i], Cn)["cn_correlation"] for i in range(len(imgsF))])
        cC = np.array([F.cn_features(imgsC[i], Cn)["cn_correlation"] for i in range(len(imgsC))])
        dF, dC = float(np.max(np.abs(stored - cF))), float(np.max(np.abs(stored - cC)))
        status = "F-match" if dF < 1e-6 else "C-match" if dC < 1e-6 else "neither"
        if status == "C-match":
            retro.append({"rel": r, "n_rows": int(len(rv)), "n_reviewed": int(rv.sum()),
                          "n_pos": int((sio.loadmat(str(sd / "labels.mat"))["labels"].ravel() == 1).sum())})
        if verbose and status != "F-match":
            print(f"  {status:8s} {r}  |stored-F| {dF:.1e} |stored-C| {dC:.1e}")
        if status == "neither":
            raise RuntimeError(f"{r}: stored cn_correlation matches neither orientation")
    if verbose:
        print(f"retro-labeled (transposed cn_correlation): {len(retro)} sessions")
    return retro


def do_build():
    BUILT.mkdir(parents=True, exist_ok=True)
    retro = do_identify(verbose=False)
    rep = {"sessions": {}, "n": len(retro)}
    for e in retro:
        r = e["rel"]
        sd = DATA_ROOT / r
        npz = np.load(sd / V1, allow_pickle=True)
        X = np.asarray(npz["feature_matrix"], float)
        assert X.shape[1] == 35, f"{r}: width {X.shape[1]}"
        A, Cn_ext, d1, d2 = load_ext(r)
        rv = reviewed_mask(npz)
        assert rv.all(), f"{r}: retro sessions have no auto-rejected rows"
        imgs = F.fcols_to_images(A, d1, d2)
        Cn = sio.loadmat(str(sd / "Cn.mat"))["Cn"]
        cn = np.array([F.cn_features(imgs[i], Cn)["cn_correlation"] for i in range(len(imgs))])
        Xc = X.copy()
        Xc[:, CN_COL] = cn
        Xc[:, 13:26] = F.compute_ranks(Xc[:, :13])
        # invariants: cols 0-11 and 26-34 byte-identical; only col 12 and the ranks move
        assert np.array_equal(Xc[:, :12], X[:, :12]) and np.array_equal(Xc[:, 26:], X[:, 26:])
        out = BUILT / f"{key(r)}.npz"
        np.savez(out, feature_matrix=Xc, feature_names=npz["feature_names"],
                 auto_rejected=npz["auto_rejected"], n_candidates=npz["n_candidates"])
        chk = np.load(out, allow_pickle=True)
        assert np.array_equal(chk["feature_matrix"], Xc) and np.array_equal(chk["auto_rejected"], npz["auto_rejected"])
        moved = float(np.mean(np.abs(cn - X[:, CN_COL]) > 0.1))
        rep["sessions"][r] = {"built": out.name, "sha256": sha256(out), "live_sha256_at_build": sha256(sd / V1),
                              "n_rows": int(len(X)), "frac_cn_moved_gt_0.1": moved,
                              "cn_p50_before": float(np.median(X[:, CN_COL])), "cn_p50_after": float(np.median(cn)),
                              "rank_rows_changed_frac": float(np.mean(np.any(np.abs(Xc[:, 13:26] - X[:, 13:26]) > 1e-12, axis=1)))}
        print(f"  built {r}: rows {len(X)}, cn p50 {rep['sessions'][r]['cn_p50_before']:.3f} -> "
              f"{rep['sessions'][r]['cn_p50_after']:.3f}, moved>0.1 {moved:.2f}")
    REPORT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {REPORT.name} ({len(rep['sessions'])} sessions)")
    return 0


def do_backup():
    rep = json.loads(REPORT.read_text())
    BK.mkdir(parents=True, exist_ok=True)
    entries = {}
    for r in rep["sessions"]:
        sd = DATA_ROOT / r
        src, dst = sd / V1, BK / f"{key(r)}__candidate_features.npz"
        h = sha256(src)
        if dst.exists():
            if sha256(dst) != h:
                print(f"ABORT: existing backup differs from live: {dst.name}")
                return 1
        else:
            shutil.copy2(str(src), str(dst))
            assert sha256(dst) == h
        entries[r] = {"backup": dst.name, "sha256": h}
    jb = BK / JOBLIB_BK_NAME
    hj = sha256(JOBLIB_LIVE)
    if jb.exists():
        if sha256(jb) != hj:
            print("ABORT: existing joblib backup differs from the live joblib")
            return 1
    else:
        shutil.copy2(str(JOBLIB_LIVE), str(jb))
        assert sha256(jb) == hj
    MANIFEST.write_text(json.dumps({"sessions": entries, "joblib_sha256": hj, "joblib_backup": JOBLIB_BK_NAME,
                                    "time": time.time()}, indent=1))
    print(f"backup OK: {len(entries)} live npz + joblib -> {BK}")
    return 0


def do_gate():
    """8-seed paired OOF, live vs corrected (corrected = built files substituted
    for the affected sessions), the deployed sqrt/4.0 recipe, via the red team's
    independent evaluator."""
    import rt_lib as rt
    rep = json.loads(REPORT.read_text())
    built = {r: BUILT / e["built"] for r, e in rep["sessions"].items()}

    def ff(sd):
        return built.get(rt.rel(sd))
    live = rt.load_pool("BLA", width=35)
    corr = rt.load_pool("BLA", width=35, feature_file_of=ff)
    res = {"n_corrected_sessions": len(built), "variants": {}}
    for label, recs in (("live", live), ("corrected", corr)):
        cv, rest, bs = rt.split_pool(recs)
        X_ag, y, g, rev, animals, names = rt.stack_agent(cv)
        X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
        oof = rt.run_oof_seeds(X_ag, y, g, X_bs, y_bs, w_bs, agent_weight=None)
        res["variants"][label] = {"oof": oof, "y": y, "rev": rev, "names": list(names), "g": g}
    y = res["variants"]["live"]["y"]
    rev = res["variants"]["live"]["rev"]
    assert np.array_equal(y, res["variants"]["corrected"]["y"])
    out = {"n_corrected_sessions": len(built), "n_pool": int(len(y)), "n_real": int(y.sum())}
    mask_c = np.isin(res["variants"]["live"]["g"], [i for i, n in enumerate(res["variants"]["live"]["names"]) if n in built])
    for label in ("live", "corrected"):
        oof = res["variants"][label]["oof"]
        tab = rt.threshold_table(oof, y, rev)
        cT, gate, det = rt.rule_T(tab)
        out[label] = {"auc_full": rt.summarize([rt.auc(y, o) for o in oof]),
                      "auc_reviewed": rt.summarize([rt.auc(y, o, rev) for o in oof]),
                      "far_at_0.04": rt.summarize([rt.far_junk(o, y, 0.04)[0] for o in oof]),
                      "junk_at_0.04": rt.summarize([rt.far_junk(o, y, 0.04)[1] for o in oof]),
                      "far_at_0.03": rt.summarize([rt.far_junk(o, y, 0.03)[0] for o in oof]),
                      "junk_at_0.03": rt.summarize([rt.far_junk(o, y, 0.03)[1] for o in oof]),
                      "auc_on_corrected_sessions": rt.summarize([rt.auc(y[mask_c], o[mask_c]) for o in oof]),
                      "rule_T": cT, "rule_detail": det, "threshold_table": tab}
        e = out[label]
        print(f"  {label:9s}: AUC full {e['auc_full']['mean']:.4f} rev {e['auc_reviewed']['mean']:.4f} | FAR@0.04 "
              f"{e['far_at_0.04']['mean']:.2f}% (max {e['far_at_0.04']['max']:.2f}) junk {e['junk_at_0.04']['mean']:.1f}% | "
              f"FAR@0.03 {e['far_at_0.03']['mean']:.2f}% (max {e['far_at_0.03']['max']:.2f}) junk {e['junk_at_0.03']['mean']:.1f}% "
              f"| on the {len(built)} sessions {e['auc_on_corrected_sessions']['mean']:.4f} | rule T {cT}")
    ol, oc = res["variants"]["live"]["oof"], res["variants"]["corrected"]["oof"]
    out["delta"] = {"full": rt.paired_delta([rt.auc(y, o) for o in ol], [rt.auc(y, o) for o in oc]),
                    "reviewed": rt.paired_delta([rt.auc(y, o, rev) for o in ol], [rt.auc(y, o, rev) for o in oc]),
                    "on_corrected_sessions": rt.paired_delta([rt.auc(y[mask_c], o[mask_c]) for o in ol],
                                                             [rt.auc(y[mask_c], o[mask_c]) for o in oc]),
                    "matched_op": {k: v for k, v in rt.matched_operating_point(ol, oc, y, 0.04).items() if k.endswith("_mean")}}
    d = out["delta"]
    print(f"  delta: full {d['full']['mean']:+.4f} ({d['full']['n_positive']}/8) reviewed {d['reviewed']['mean']:+.4f} | "
          f"on corrected sessions {d['on_corrected_sessions']['mean']:+.4f} ({d['on_corrected_sessions']['n_positive']}/8) | "
          f"junk at matched FAR {d['matched_op']['junk_ref_mean']:.1f}% -> {d['matched_op']['junk_new_at_matched_far_mean']:.1f}%")
    out["pin_hash"] = rt.pin_hash()
    GATE.write_text(json.dumps(out, indent=1, default=rt._jsonable))
    print(f"wrote {GATE.name}")
    return 0


def _bla_watcher_running():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like '*python*' -and "
          f"$_.ProcessId -ne {os.getpid()} }} | Select-Object -ExpandProperty CommandLine")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return any(("watcher.py" in l and "watcher_" not in l) for l in (r.stdout or "").splitlines())
    except Exception:
        pass
    return None


def do_preflight(strict=True):
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'ok   ' if c else 'FAIL '} {m}")
    print("preflight")
    chk(REPORT.exists(), "cnfix_report.json present (build ran)")
    chk(MANIFEST.exists(), "backup manifest present")
    chk(GATE.exists(), "gate ran (cnfix_gate.json)")
    running = _bla_watcher_running()
    chk(running is False, f"BLA watcher not running ({'UNKNOWN' if running is None else running})")
    log = AGENT / "logs" / "watcher.log"
    age = (time.time() - log.stat().st_mtime) / 60 if log.exists() else None
    chk(age is None or age > 2, f"watcher.log quiet ({age:.0f} min)" if age else "no watcher.log")
    if REPORT.exists() and MANIFEST.exists():
        rep, man = json.loads(REPORT.read_text()), json.loads(MANIFEST.read_text())
        drift = [r for r, e in rep["sessions"].items() if sha256(DATA_ROOT / r / V1) != e["live_sha256_at_build"]]
        chk(not drift, f"live files unchanged since build ({len(drift)} drifted)")
        bad = [r for r, e in rep["sessions"].items() if sha256(BUILT / e["built"]) != e["sha256"]]
        chk(not bad, f"built files match the report ({len(bad)} differ)")
        chk(set(rep["sessions"]) == set(man["sessions"]), "backup covers exactly the built sessions")
        chk(sha256(JOBLIB_LIVE) == man["joblib_sha256"], "live joblib unchanged since backup")
    print(f"PREFLIGHT: {'PASS' if ok else 'NOT READY'}")
    return 0 if ok else 1


def do_swap(freeze):
    if not freeze:
        print("refusing: `swap` needs --freeze")
        return 2
    if do_preflight(strict=True) != 0:
        print("ABORT: preflight failed")
        return 1
    rep = json.loads(REPORT.read_text())
    out = {}
    for r, e in rep["sessions"].items():
        sd = DATA_ROOT / r
        src = BUILT / e["built"]
        tmp = sd / (V1 + ".cnfix_tmp")
        shutil.copy2(str(src), str(tmp))
        if sha256(tmp) != e["sha256"]:
            tmp.unlink()
            print(f"ABORT: copy verify failed for {r}")
            return 1
        os.replace(str(tmp), str(sd / V1))
        if sha256(sd / V1) != e["sha256"]:
            print(f"ABORT mid-swap: {r}")
            return 1
        out[r] = {"sha256": e["sha256"]}
    SWAP_REPORT.write_text(json.dumps(out, indent=1))
    print(f"SWAP DONE: {len(out)} sessions refreshed. Next: retrain BLA, then `verify`.")
    return 0


def do_verify(T):
    ok = True

    def chk(c, m):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'ok   ' if c else 'FAIL '} {m}")
    m = joblib.load(str(JOBLIB_LIVE))
    print("1. joblib contract")
    chk(m.get("model_type") == "xgboost", f"model_type = {m.get('model_type')}")
    chk(m["scaler"].n_features_in_ == 35, f"scaler width = {m['scaler'].n_features_in_}")
    chk("first_pass_scaler" in m and m["first_pass_scaler"].n_features_in_ == 13, "companion first-pass at width 13")
    chk(abs(m.get("reject_threshold", -1) - T) < 1e-9, f"reject_threshold = {m.get('reject_threshold')} (decided {T})")
    chk(m.get("feature_version") == 2, f"feature_version = {m.get('feature_version')}")
    chk(m.get("n_sessions") == 170, f"n_sessions = {m.get('n_sessions')} (expected 170)")
    chk(m.get("n_excluded_ambiguous") == 4494, f"n_excluded_ambiguous = {m.get('n_excluded_ambiguous')} (expected 4494)")
    print("2. corpus identity")
    rep, man = json.loads(REPORT.read_text()), json.loads(MANIFEST.read_text())
    n_sha = n_id = 0
    for r, e in rep["sessions"].items():
        sd = DATA_ROOT / r
        if sha256(sd / V1) == e["sha256"]:
            n_sha += 1
        live = np.load(sd / V1, allow_pickle=True)
        back = np.load(BK / man["sessions"][r]["backup"], allow_pickle=True)
        Xl, Xb = live["feature_matrix"], back["feature_matrix"]
        if (np.array_equal(Xl[:, :12], Xb[:, :12]) and np.array_equal(Xl[:, 26:], Xb[:, 26:])
                and np.array_equal(live["auto_rejected"], back["auto_rejected"])
                and live["n_candidates"][0] == back["n_candidates"][0]
                and not np.array_equal(Xl[:, 12], Xb[:, 12])):
            n_id += 1
    chk(n_sha == len(rep["sessions"]), f"live files == built: {n_sha}/{len(rep['sessions'])}")
    chk(n_id == len(rep["sessions"]), f"cols 0-11 + 26-34 + auto_rejected preserved, col 12 changed: {n_id}/{len(rep['sessions'])}")
    print("3. deployed smoke (bla21 autopsy cells)")
    npz = np.load(DATA_ROOT / SMOKE_REL / V1, allow_pickle=True)
    s = m["clf"].predict_proba(m["scaler"].transform(npz["feature_matrix"]))[:, 1]
    for ci in SMOKE_IDX:
        chk(s[ci] > T, f"'Neuron {ci + 1}' deployed score {s[ci]:.3f} vs T={T:.2f} (in-sample)")
    print("4. watcher retrain trigger")
    newest = max((sd / "labels.mat").stat().st_mtime for td in DATA_ROOT.iterdir() if td.is_dir() and not td.name.startswith(".")
                 for sd in td.iterdir() if sd.is_dir() and (sd / "labels.mat").exists())
    chk(JOBLIB_LIVE.stat().st_mtime > newest, f"joblib newer than every labels.mat ({JOBLIB_LIVE.stat().st_mtime - newest:+.0f} s)")
    print(f"VERIFY: {'ALL PASS' if ok else 'FAILURES -- do not restart the watcher'}")
    return 0 if ok else 1


def do_rollback():
    man = json.loads(MANIFEST.read_text())
    t_bk = man["time"]
    stale = [r for r in man["sessions"] if (DATA_ROOT / r / "labels.mat").stat().st_mtime > t_bk]
    if stale:
        print(f"REFUSED: {len(stale)} session(s) have labels newer than the backup: {stale}")
        return 1
    for r, e in man["sessions"].items():
        shutil.copy2(str(BK / e["backup"]), str(DATA_ROOT / r / V1))
        assert sha256(DATA_ROOT / r / V1) == e["sha256"]
    shutil.copy2(str(BK / man["joblib_backup"]), str(JOBLIB_LIVE))
    assert sha256(JOBLIB_LIVE) == man["joblib_sha256"]
    print(f"ROLLBACK DONE: {len(man['sessions'])} npz + joblib restored byte-exact. Also git-revert the code fix if the "
          f"transposed behaviour must return (it should not).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["identify", "build", "backup", "gate", "preflight", "swap", "verify", "rollback"])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.04, help="verify: the deployed T")
    args = ap.parse_args()
    sys.exit({"identify": lambda: (do_identify(), 0)[1], "build": do_build, "backup": do_backup, "gate": do_gate,
              "preflight": do_preflight, "swap": lambda: do_swap(args.freeze),
              "verify": lambda: do_verify(args.threshold), "rollback": do_rollback}[args.cmd]())
