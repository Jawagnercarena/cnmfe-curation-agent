"""Temp-dir test for prune_movie_mats.py.

Builds a fake DATA_PARENT (one session per keep rule, plus deletable ones), a fake
retro data tree and a fake training cache, all under one temp folder, then runs a
dry run and an apply.  Every file in the tree is hashed before and after: the dry
run must change nothing but its manifest, and the apply must remove exactly the
expected movies.  The real data tree and the real retro project are never touched.
"""
import hashlib
import io
import os
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import numpy as np
import tifffile

import movie_mat
import prune_movie_mats as pm
import training_common as tc

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def write_movie(path: Path, fr: np.ndarray):
    T, d1, d2 = fr.shape
    with h5py.File(str(path), "w") as h:
        h["Y"] = fr.transpose(0, 2, 1)
        h["Ysiz"] = np.array([[d1, d2, T]], dtype=float)


def name(i, animal="bla50"):
    return f"AVG5x-TSeries-0101{i:02d}-{animal}-500um-3z-000"


OLD = time.time() - 3 * 86400


def session(task_dir: Path, nm: str, labels=True, tif=True, movie=True, good=True, seed=0,
            extra=(), old=True):
    sd = task_dir / nm
    sd.mkdir(parents=True)
    fr = np.random.default_rng(seed).integers(0, 255, size=(20, 8, 12)).astype(np.uint8)
    if tif:
        tifffile.imwrite(str(sd / f"{nm}.tif"), fr)
    if movie:
        mv = fr.copy()
        if not good:
            mv[0, 0, 0] ^= 1
        write_movie(sd / f"{nm}.mat", mv)
        if old:
            os.utime(sd / f"{nm}.mat", (OLD, OLD))
    files = ["neuron.mat", "ROIs.jpg", "Cn.mat", "spatial_footprints.mat", "candidate_features.npz"]
    if labels:
        files.append("labels.mat")
    for f in list(files) + list(extra):
        (sd / f).write_bytes(b"x" * 64)
    return sd


def tree_hashes(root: Path, skip_manifest=True):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            if skip_manifest and rel.parts[0] == pm.MANIFEST_DIRNAME:
                continue
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


tmp = Path(tempfile.mkdtemp(prefix="prune_movies_"))
try:
    data = tmp / "data"
    t1 = data / "BLA" / "t1"
    ok1 = session(t1, name(1), seed=1)
    ok2 = session(t1, name(2), seed=2)
    unfin = session(t1, name(3), labels=False)
    notif = session(t1, name(4), tif=False)
    bad = session(t1, name(5), good=False)
    pend = session(t1, name(6), extra=["ROIs_candidates.jpg", "review_neuron.mat"])
    (pend / "ROIs.jpg").unlink()
    fresh = session(t1, name(7), old=False)
    train = session(t1, name(8), extra=[tc.TRAINING_MARKER])
    other = session(t1, name(9), seed=9)
    write_movie(other / "AVG5x-TSeries-0101-bla50-um-3z-000.mat", np.zeros((20, 8, 12), np.uint8))
    partial = session(t1, name(10), seed=10)
    (partial / ".movie_rebuild").mkdir()
    (partial / ".movie_rebuild" / f"{partial.name}.mat").write_bytes(b"partial")
    inuse = session(t1, name(11), seed=11)
    retro = session(data / "vCA1" / "t2", name(12, "pnb98"), seed=12)
    hidden = session(data / "BLA" / ".excluded" / "t1", name(13), seed=13)
    nolab_nomovie = session(t1, name(14), movie=False)

    cache_ok = tc.movie_cache_dir("BLA", "t1", ok1.name, data / ".training") / f"{ok1.name}.mat"
    cache_ok.parent.mkdir(parents=True)
    shutil.copy2(str(ok1 / f"{ok1.name}.mat"), str(cache_ok))
    cache_gone = tc.movie_cache_dir("BLA", "t1", name(20), data / ".training") / f"{name(20)}.mat"
    cache_gone.parent.mkdir(parents=True)
    shutil.copy2(str(ok1 / f"{ok1.name}.mat"), str(cache_gone))

    retro_root = tmp / "retro"
    (retro_root / "data" / "pnb98").mkdir(parents=True)
    retro_results = retro_root / "results"
    pm.OTHER_MOVIE_BYTES = 1000        # test movies (~4 KB) count as big, 64-byte outputs do not

    def go(**kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows, man = pm.run(data, retro_root=retro_results, **kw)
        return {r["path"]: r for r in rows}, man, buf.getvalue()

    # ---------------------------------------------------------------- guards ----
    print("guards")
    check(not pm.is_local_dir(Path(r"\\server\share\Julian_CNMFe")), "UNC path refused")
    check(pm.is_local_dir(tmp), "local temp folder accepted")
    if Path("X:/").exists():
        check(not pm.is_local_dir(Path("X:/")), "mapped network drive X: refused")
    else:
        print("  (X: not mapped here; mapped-drive refusal not exercised)")

    def rel(sd):
        return str(sd.relative_to(data) / f"{sd.name}.mat")

    # ---------------------------------------------------------------- dry run ---
    print("dry run")
    before = tree_hashes(data)
    rows, man, out = go()
    check(tree_hashes(data) == before, "dry run changes nothing in the tree (manifest aside)")
    check(man.exists() and man.parent.name == pm.MANIFEST_DIRNAME and "dryrun_" in man.name,
          "dry run wrote its manifest under .movie_prune/")
    want = {
        rel(ok1): ("would delete", ""), rel(ok2): ("would delete", ""),
        rel(other): ("would delete", ""), rel(partial): ("would delete", ""),
        rel(inuse): ("would delete", ""),
        rel(unfin): ("keep", "not finished"), rel(notif): ("keep", "no AVG5x"),
        rel(bad): ("keep", "tif does not reproduce"), rel(pend): ("keep", "watcher would pre-compute"),
        rel(fresh): ("keep", "younger than"), rel(train): ("keep", "training sandbox"),
        rel(retro): ("keep", "retro-tracked animal pnb98"),
        str(cache_ok.relative_to(data)): ("would delete", ""),
        str(cache_gone.relative_to(data)): ("keep", "production session has no .tif"),
        str(other.relative_to(data) / "AVG5x-TSeries-0101-bla50-um-3z-000.mat"): ("keep", "another name"),
        str(partial.relative_to(data) / ".movie_rebuild"): ("keep", "interrupted rebuild"),
    }
    for path, (dec, why) in want.items():
        r = rows.get(path)
        check(r is not None and r["decision"] == dec and why in r["reason"],
              f"{dec:<12} {why or '(all checks pass)'}: {Path(path).name if r else path}"
              + ("" if r and r["decision"] == dec and why in r["reason"] else f"  -> got {r}"))
    check(set(rows) == set(want), f"exactly the expected rows (extra: {set(rows) - set(want)}, "
                                  f"missing: {set(want) - set(rows)})")
    check(not any(str(hidden.name) in p for p in rows), "dot folder (.excluded) never scanned")
    check(not any(nolab_nomovie.name in p for p in rows), "session without a movie not listed")

    rows_r, _, _ = go(include_retro=True)
    check(rows_r[rel(retro)]["decision"] == "would delete", "--include-retro makes the retro movie a candidate")
    rows_l, _, _ = go(limit=2)
    n_would = sum(r["decision"] == "would delete" for r in rows_l.values())
    n_beyond = sum("beyond --limit" in r["reason"] for r in rows_l.values())
    # past the limit nothing is verified, so the bad movie is 'beyond --limit' too
    check(n_would == 2 and n_beyond == 5,
          f"--limit 2: two candidates, the other five unverified 'beyond --limit' "
          f"(got {n_would} / {n_beyond})")
    rows_s, _, _ = go(sessions=[f"BLA\\t1\\{ok2.name}"])
    check(set(rows_s) == {rel(ok2)}, "--session restricts the run to that session")

    # ---------------------------------------------------------------- apply -----
    print("apply")
    before = tree_hashes(data)
    with h5py.File(str(inuse / f"{inuse.name}.mat"), "r"):
        rows, man, out = go(apply=True)
    after = tree_hashes(data)
    gone = set(before) - set(after)
    expect_gone = {rel(ok1), rel(ok2), rel(other), rel(partial), str(cache_ok.relative_to(data))}
    check(gone == expect_gone, f"apply removed exactly the expected movies (extra: {gone - expect_gone}, "
                               f"kept: {expect_gone - gone})")
    check(all(after[k] == v for k, v in before.items() if k not in gone),
          "every other file is byte-identical (tifs, outputs, other-name movie, partial, held-open movie)")
    check(not (set(after) - set(before)), "apply created no file in the tree (manifest aside)")
    check(rows[rel(inuse)]["decision"] == "keep" and "could not delete" in rows[rel(inuse)]["reason"],
          f"a movie held open by another reader is reported, not deleted: {rows[rel(inuse)]['reason'][:60]}")
    check(all(d.is_dir() for d in (ok1, ok2, other, partial)), "no folder removed")
    check("apply_" in man.name and sum(r["decision"] == "deleted" for r in rows.values()) == 5,
          "apply manifest records 5 deletions")

    pm.RETRO_RESULTS = str(retro_results)       # main() reads the module default
    with h5py.File(str(inuse / f"{inuse.name}.mat"), "r"):
        with redirect_stdout(io.StringIO()):
            rc = pm.main(["--data-parent", str(data), "--apply", "--session", f"BLA\\t1\\{inuse.name}"])
    check(rc == 1, f"main: a delete that fails -> exit 1 (rc={rc})")
    with redirect_stdout(io.StringIO()):
        rc = pm.main(["--data-parent", str(data), "--apply"])
    left = {str(p.relative_to(data)) for p in data.rglob("*.mat") if p.name == f"{p.parent.name}.mat"}
    check(rc == 0 and not (inuse / f"{inuse.name}.mat").exists(),
          f"main: second apply (file released) deletes it, exit 0 (rc={rc})")
    check(left == {rel(notif), rel(bad), rel(pend), rel(fresh), rel(train), rel(retro),
                   rel(unfin), str(cache_gone.relative_to(data)),
                   str(hidden.relative_to(data) / f"{hidden.name}.mat")},
          f"after both applies only the kept movies remain ({len(left)})")
    check((partial / ".movie_rebuild" / f"{partial.name}.mat").exists(),
          "the interrupted-rebuild leftover is still there (listed, never deleted)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
