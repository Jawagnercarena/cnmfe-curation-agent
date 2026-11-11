"""Temp-dir test for movie rebuild-on-demand (movie_mat.py, rebuild_movie_mat.m, the
CNMFe_final_save.m locate block) and its callers (push_review_bundle.collect_files,
rereview.py request, push_training_bundle.py).

Everything lives under one temp folder: fake sessions with tiny synthetic tifs, a
fake .training root and a fake exchange.  The real data tree and the real exchange
are never touched.  Parts 1-3 stub the MATLAB rebuild; part 4 runs real MATLAB
(three short -batch calls) on the tiny tifs.  Pass --no-matlab to skip part 4.
"""
import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import numpy as np
import scipy.io as sio
import tifffile

import movie_mat
import push_review_bundle as P
import push_training_bundle as pb
import rereview as rr
import run_cnmfe
import training_common as tc
from local_config import REPO_ROOT

NO_MATLAB = "--no-matlab" in sys.argv
fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def run(fn, *a, **kw):
    buf = io.StringIO()
    rc = None
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = fn(*a, **kw)
    except SystemExit as e:
        rc = e.code
        if isinstance(e.code, str):
            buf.write(e.code)
            rc = 1
    return rc, buf.getvalue()


def frames(T=40, d1=16, d2=24, dtype=np.uint8, seed=0):
    rng = np.random.default_rng(seed)
    hi = 255 if dtype == np.uint8 else 4000
    return rng.integers(0, hi, size=(T, d1, d2)).astype(dtype)


def write_movie(path: Path, fr: np.ndarray):
    """A movie .mat laid out as MATLAB -v7.3 stores Y (d1 x d2 x T) and Ysiz (3 x 1)."""
    T, d1, d2 = fr.shape
    with h5py.File(str(path), "w") as h:
        h["Y"] = fr.transpose(0, 2, 1)
        h["Ysiz"] = np.array([[d1, d2, T]], dtype=float)


def make_session(root: Path, name: str, fr: np.ndarray, movie=True, tif=True, imagej=False):
    sd = root / name
    sd.mkdir(parents=True)
    if tif:
        tifffile.imwrite(str(sd / f"{name}.tif"), fr, imagej=imagej)
    if movie:
        write_movie(sd / f"{name}.mat", fr)
    for fname, size in [("review_neuron.mat", 3000), ("Cn.mat", 200), ("pnr.mat", 100),
                        ("Ybg_weights.mat", 500), ("review_report.pdf", 400),
                        ("review_summary.txt", 50), ("ROIs_candidates.jpg", 80)]:
        (sd / fname).write_bytes(b"x" * size)
    return sd


class FakeRebuild:
    """Stands in for movie_mat.rebuild: records calls, writes a real movie from the tif."""
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, session_dir, log=print, out_dir=None):
        self.calls.append((Path(session_dir), out_dir))
        if self.fail:
            raise RuntimeError("rebuild failed (fake)")
        target = movie_mat._target(session_dir, out_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_movie(target, tifffile.imread(str(movie_mat.movie_tif(session_dir))))
        return target


real_rebuild = movie_mat.rebuild
tmp = Path(tempfile.mkdtemp(prefix="movie_mat_"))
try:
    # ---------------------------------------------------------------- 1 ----
    print("1. verify_against_tif")
    vd = tmp / "verify"
    vd.mkdir()
    fr = frames(T=40)
    tifffile.imwrite(str(vd / "a.tif"), fr)
    write_movie(vd / "a.mat", fr)
    ok, d = movie_mat.verify_against_tif(vd / "a.mat", vd / "a.tif")
    check(ok and "32/32 sampled" in d, f"identical movie verifies (sampled): {d}")
    ok, d = movie_mat.verify_against_tif(vd / "a.mat", vd / "a.tif", full=True)
    check(ok and "all 40 frames" in d, f"identical movie verifies (full): {d}")

    tifffile.imwrite(str(vd / "ij.tif"), fr, imagej=True)
    ok, d = movie_mat.verify_against_tif(vd / "a.mat", vd / "ij.tif", full=True)
    check(ok, f"ImageJ tif (memory-mapped reader) verifies: {d}")
    tifffile.imwrite(str(vd / "z.tif"), fr, compression="zlib")
    ok, d = movie_mat.verify_against_tif(vd / "a.mat", vd / "z.tif", full=True)
    check(ok, f"compressed tif (page reader) verifies: {d}")

    sampled = set(np.linspace(0, 39, 32).round().astype(int).tolist())
    unsampled = min(set(range(40)) - sampled)
    bad = fr.copy()
    bad[unsampled, 3, 3] ^= 1
    write_movie(vd / "b.mat", bad)
    ok_s, _ = movie_mat.verify_against_tif(vd / "b.mat", vd / "a.tif")
    ok_f, d = movie_mat.verify_against_tif(vd / "b.mat", vd / "a.tif", full=True)
    check(ok_s and not ok_f and f"frame {unsampled}" in d,
          f"a change in an unsampled frame passes the sample, fails --full ({d})")
    bad = fr.copy()
    bad[0, 0, 0] ^= 1
    write_movie(vd / "c.mat", bad)
    ok, d = movie_mat.verify_against_tif(vd / "c.mat", vd / "a.tif")
    check(not ok and "frame 0" in d, f"first frame differs -> not verified ({d})")
    write_movie(vd / "d.mat", fr[:39])
    ok, d = movie_mat.verify_against_tif(vd / "d.mat", vd / "a.tif")
    check(not ok and "39" in d, f"frame count differs -> not verified ({d})")
    write_movie(vd / "e.mat", fr.astype(np.uint16))
    ok, d = movie_mat.verify_against_tif(vd / "e.mat", vd / "a.tif")
    check(not ok and "pixel type" in d, f"pixel type differs -> not verified ({d})")
    (vd / "f.mat").write_bytes(b"MATLAB 5.0 MAT-file, not HDF5" + b"\0" * 200)
    ok, d = movie_mat.verify_against_tif(vd / "f.mat", vd / "a.tif")
    check(not ok, f"non-HDF5 .mat -> not verified, no exception ({d})")
    ok, d = movie_mat.verify_against_tif(vd / "a.mat", vd / "missing.tif")
    check(not ok, f"missing tif -> not verified, no exception ({d})")

    # ---------------------------------------------------------------- 2 ----
    print("2. push_review_bundle.collect_files / ensure (MATLAB stubbed)")
    data = tmp / "data"
    task = data / "BLA" / "task1"
    s_ok = make_session(task, "AVG5x-TSeries-010126-bla98-500um-3z-000", fr, movie=False)
    fake = FakeRebuild()
    movie_mat.rebuild = fake
    files, out = run(P.collect_files, s_ok, dry_run=True)
    check("would rebuild" in out and not fake.calls and not movie_mat.movie_mat(s_ok).exists(),
          "dry run: says it would rebuild, runs nothing, writes nothing")
    check(files is not None and movie_mat.movie_mat(s_ok) not in files and "WARNING" not in out,
          "dry run: movie not in the file list, no missing-video warning")
    files, out = run(P.collect_files, s_ok)
    check(len(fake.calls) == 1 and files[0] == movie_mat.movie_mat(s_ok),
          "real run: rebuilt once, movie first in the bundle list")
    files, out = run(P.collect_files, s_ok)
    check(len(fake.calls) == 1 and files[0] == movie_mat.movie_mat(s_ok),
          "existing movie: no second rebuild")

    s_fail = make_session(task, "AVG5x-TSeries-010226-bla98-500um-3z-000", fr, movie=False)
    movie_mat.rebuild = FakeRebuild(fail=True)
    try:
        with redirect_stdout(io.StringIO()):
            P.collect_files(s_fail)
        check(False, "failed rebuild raises (no bundle without video)")
    except RuntimeError as e:
        check("rebuild failed" in str(e), "failed rebuild raises (no bundle without video)")

    s_noreq = make_session(task, "AVG5x-TSeries-010326-bla98-500um-3z-000", fr, movie=False)
    (s_noreq / "review_neuron.mat").unlink()
    fake = FakeRebuild()
    movie_mat.rebuild = fake
    try:
        with redirect_stdout(io.StringIO()):
            P.collect_files(s_noreq)
        check(False, "missing review_neuron.mat raises before any rebuild")
    except ValueError:
        check(not fake.calls, "missing review_neuron.mat raises before any rebuild")

    s_none = make_session(task, "AVG5x-TSeries-010426-bla98-500um-3z-000", fr, movie=False, tif=False)
    files, out = run(P.collect_files, s_none)
    check(not fake.calls and "WARNING: raw video" in out and len(files) > 0,
          "no movie and no tif: old behaviour (warn, push without video)")

    # ---------------------------------------------------------------- 3 ----
    print("3. rereview.py request and push_training_bundle.py (MATLAB stubbed)")
    exchange = tmp / "exchange"
    (exchange / "outbox").mkdir(parents=True)
    (exchange / "inbox").mkdir()
    (exchange / "training").mkdir()
    rs = make_session(task, "AVG5x-TSeries-010526-bla97-500um-3z-000", fr, movie=False)
    N = 10
    rng = np.random.default_rng(0)
    np.savez(rs / "candidate_features.npz", feature_matrix=rng.normal(size=(N, 35)),
             feature_names=np.array([f"f{i}" for i in range(35)]),
             auto_rejected=np.array([1, 2], dtype=int), n_candidates=np.array([N]))
    sio.savemat(str(rs / "labels.mat"), {"labels": np.ones((N - 2, 1)),
                                          "motion_delete": np.zeros((N - 2, 1))})
    (rs / "labels_provenance.txt").write_text("reviewer: Taylor\ningested: 2026-09-01 10:00:00\nsource: x\n")
    for nm in ("neuron.mat", "ROIs.jpg"):
        (rs / nm).write_bytes(b"out")
    fake = FakeRebuild()
    movie_mat.rebuild = fake
    base = ["--data-parent", str(data), "request", "BLA\\task1\\" + rs.name,
            "--assignee", "Aneesh", "--reason", "test", "--exchange", str(exchange)]
    rc, out = run(rr.main, base + ["--dry-run"])
    check(rc == 0 and "would rebuild" in out and not fake.calls
          and not movie_mat.movie_mat(rs).exists(),
          f"rereview request --dry-run: says it would rebuild, rebuilds nothing (rc={rc})")
    rc, out = run(rr.main, base)
    bundle = exchange / "outbox" / "Aneesh" / "BLA" / "task1" / rs.name
    check(rc == 0 and len(fake.calls) == 1 and (bundle / f"{rs.name}.mat").exists(),
          f"rereview request: movie rebuilt in the session folder and staged (rc={rc})")
    check(fake.calls[0][1] is None, "rereview rebuilds into the session folder itself")

    troot = tmp / ".training"
    ts = make_session(data / "BLA" / "task2", "AVG5x-TSeries-010626-bla96-500um-3z-000", fr, movie=False)
    kp = tc.key_path("BLA", "task2", ts.name, troot)
    kp.parent.mkdir(parents=True)
    sio.savemat(str(kp), {"reference_reviewer": "Taylor", "n_review": 5.0})
    fake = FakeRebuild()
    movie_mat.rebuild = fake
    targs = [str(ts), "--trainee", "Alice", "--stage", "1", "--exchange", str(exchange),
             "--training-root", str(troot)]
    rc, out = run(pb.main, targs + ["--dry-run"])
    check(rc == 0 and "would rebuild" in out and not fake.calls, f"training push --dry-run: no rebuild (rc={rc})")
    before = sorted(p.name for p in ts.iterdir())
    rc, out = run(pb.main, targs)
    cache = tc.movie_cache_dir("BLA", "task2", ts.name, troot)
    dest = exchange / "training" / "Alice" / "BLA" / "task2" / ts.name
    check(rc == 0 and len(fake.calls) == 1 and (cache / f"{ts.name}.mat").exists()
          and (dest / f"{ts.name}.mat").exists(),
          f"training push: movie rebuilt into the training cache and staged (rc={rc})")
    check(sorted(p.name for p in ts.iterdir()) == before,
          "training push wrote nothing into the production session folder")
    marker = tc.read_training_marker(dest / tc.TRAINING_MARKER)
    check(marker.get("has_video") == "1", "training marker has_video = 1")
    rc, out = run(pb.main, targs[:2] + ["Bob"] + targs[3:])
    check(rc == 0 and len(fake.calls) == 1 and "using the rebuilt movie" in out,
          "second trainee: cached movie reused, no second rebuild")
    movie_mat.rebuild = real_rebuild

    # ---------------------------------------------------------------- 4 ----
    if NO_MATLAB:
        print("4. real MATLAB: SKIPPED (--no-matlab)")
    else:
        print("4. real MATLAB (rebuild_movie_mat.m, tif2mat agreement, final_save block)")
        msgs = []
        log = msgs.append
        fr8 = frames(T=50, d1=32, d2=40, seed=1)
        rs1 = make_session(data / "vCA1" / "t", "AVG5x-TSeries-020126-bla90-400um-3z-000", fr8, movie=False)
        files, out = run(P.collect_files, rs1)
        mat1 = movie_mat.movie_mat(rs1)
        ok, d = movie_mat.verify_against_tif(mat1, movie_mat.movie_tif(rs1), full=True) if mat1.exists() else (False, "missing")
        check(mat1.exists() and files[0] == mat1 and ok,
              f"collect_files with real MATLAB: rebuilt and verified ({d})")
        check(not (rs1 / ".movie_rebuild").exists(), "temp .movie_rebuild folder removed after success")

        fr16 = frames(T=30, d1=24, d2=20, dtype=np.uint16, seed=2)
        rs2 = make_session(data / "vCA1" / "t", "TSeries-020226-DG9-300um-2z-000A", fr16, movie=False, imagej=True)
        ref = tmp / "tif2mat_ref"
        ref.mkdir()
        shutil.copy2(str(movie_mat.movie_tif(rs1)), str(ref / "r8.tif"))
        shutil.copy2(str(movie_mat.movie_tif(rs2)), str(ref / "r16.tif"))
        lib = (Path(REPO_ROOT) / "ca_source_extraction").as_posix()
        script = (f"addpath(genpath('{lib}')); "
                  f"tif2mat('{(ref / 'r8.tif').as_posix()}'); tif2mat('{(ref / 'r16.tif').as_posix()}'); "
                  f"rebuild_movie_mat('{movie_mat.movie_tif(rs2).as_posix()}', '{movie_mat.movie_mat(rs2).as_posix()}'); "
                  f"try; rebuild_movie_mat('{movie_mat.movie_tif(rs2).as_posix()}', '{movie_mat.movie_mat(rs2).as_posix()}'); "
                  f"disp('OVERWRITE_ALLOWED'); catch e; disp(['REFUSED ' e.identifier]); end")
        ok = run_cnmfe._run_matlab(script, log, timeout_hours=0.3)
        text = "\n".join(msgs)
        check(ok and "REFUSED rebuild_movie_mat:exists" in text and "OVERWRITE_ALLOWED" not in text,
              "rebuild_movie_mat refuses to overwrite an existing movie")
        for ref_mat, mine in ((ref / "r8.mat", mat1), (ref / "r16.mat", movie_mat.movie_mat(rs2))):
            with h5py.File(str(ref_mat), "r") as a, h5py.File(str(mine), "r") as b:
                same = (np.array_equal(a["Y"][()], b["Y"][()]) and a["Y"].dtype == b["Y"].dtype
                        and np.array_equal(a["Ysiz"][()], b["Ysiz"][()])
                        and a["Ysiz"].dtype == b["Ysiz"].dtype)
            check(same, f"{mine.name}: Y, Ysiz and their types identical to tif2mat's output")
        ok, d = movie_mat.verify_against_tif(movie_mat.movie_mat(rs2), movie_mat.movie_tif(rs2), full=True)
        check(ok, f"ImageJ uint16 tif rebuilt and verified ({d})")

        # the real CNMFe_final_save.m block, cut from the file, run on a movie-less session
        src = (Path(REPO_ROOT) / "CNMFe_final_save.m").read_text(encoding="utf-8", errors="replace").splitlines()
        i0 = next(i for i, l in enumerate(src) if l.startswith("% Locate the .mat data file"))
        i1 = next(i for i, l in enumerate(src) if l.startswith("d1 = Ysiz(1); d2 = Ysiz(2); numFrame = Ysiz(3);"))
        fs = make_session(data / "vCA1" / "t", "AVG5x-TSeries-020326-bla91-410um-3z-000", fr8, movie=False)
        drv = tmp / "fs_block"
        drv.mkdir()
        body = "\n".join(src[i0:i1 + 1])
        for name, sdir in (("blk_rebuild.m", fs), ("blk_present.m", fs)):
            (drv / name).write_text(
                f"addpath(genpath('{lib}'));\nsession_dir = '{sdir.as_posix()}';\n"
                f"[~, session_nm] = fileparts(session_dir);\n{body}\n"
                f"fprintf('BLOCK_OK %d %d %d %s\\n', d1, d2, numFrame, nam_mat);\n", encoding="ascii")
        msgs.clear()
        ok = run_cnmfe._run_matlab(f"run('{(drv / 'blk_rebuild.m').as_posix()}'); "
                                   f"run('{(drv / 'blk_present.m').as_posix()}');", log, timeout_hours=0.3)
        text = "\n".join(msgs)
        n_rebuild = text.count("rebuilding it from the .tif")
        check(ok and text.count("BLOCK_OK 32 40 50") == 2 and n_rebuild == 1,
              f"final_save block: rebuilds when missing, reuses when present (rebuild msgs={n_rebuild})")
        ok, d = movie_mat.verify_against_tif(movie_mat.movie_mat(fs), movie_mat.movie_tif(fs), full=True)
        check(ok, f"final_save block's rebuilt movie verifies ({d})")
finally:
    movie_mat.rebuild = real_rebuild
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
