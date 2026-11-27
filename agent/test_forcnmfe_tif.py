"""Temp-dir test for the session .tif copy into the shared for-CNMFe folder
(push_review_bundle.stage_tif_forcnmfe, called at the end of stage_one).

Everything lives under one temp folder: a fake data tree, a fake for-CNMFe root
and a fake exchange.  DATA_PARENT, RETRO_RESULTS and FORCNMFE_ROOT are pointed at
the temp tree (RETRO_RESULTS empty, so no retro hook runs), so the real server,
the real data tree and the retro project are never touched.  The real
config/forcnmfe_folders.json is only read.
"""
import io
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import push_review_bundle as P

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = fn(*a, **kw)
    return res, buf.getvalue()


def make_session(task_dir: Path, name: str, tif=b"TIF-" * 1000, ext=".tif"):
    sd = task_dir / name
    sd.mkdir(parents=True)
    for fname, size in [("review_neuron.mat", 3000), ("Cn.mat", 200), ("pnr.mat", 100),
                        ("Ybg_weights.mat", 500), ("review_summary.txt", 50),
                        ("ROIs_candidates.jpg", 80), (f"{name}.mat", 10000)]:
        (sd / fname).write_bytes(b"x" * size)
    if tif is not None:
        (sd / f"{name}{ext}").write_bytes(tif)
    return sd


def tree(root: Path):
    """Every file and folder under root with its bytes (files) -- to prove what changed."""
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


tmp = Path(tempfile.mkdtemp(prefix="forcnmfe_tif_"))
try:
    data = tmp / "data"
    srv = tmp / "for CNMFe"
    (srv / "6odorDualDiffRew").mkdir(parents=True)
    (srv / "bla pnb 3odor" / "cohort4").mkdir(parents=True)
    (srv / "bla pnb 3odor" / "cohort5").mkdir(parents=True)
    exchange = tmp / "exchange"
    (exchange / "outbox").mkdir(parents=True)
    P.DATA_PARENT = data
    P.RETRO_RESULTS = ""                  # no retro hook, no real retro tree
    P.FORCNMFE_ROOT = str(srv)            # never the real server
    m = {"vCA1/6odorDualDiffRew": {"*": "6odorDualDiffRew"},
         "vCA1/3odor": {"pnb98": "bla pnb 3odor/cohort4", "pnb99": "bla pnb 3odor/cohort5"},
         "vCA1/CTA": {"*": "pnb CTA"}}          # pnb CTA does not exist on the fake server
    six = data / "vCA1" / "6odorDualDiffRew"
    odor3 = data / "vCA1" / "3odor"

    print("1. off / nothing to copy / server unreachable")
    sd = make_session(six, "AVG5x-TSeries-090526-pnb99-330um-3z-000")
    before = tree(srv)
    line = P.stage_tif_forcnmfe(sd, False, root="", mapping=m)
    check("CNMFE_FORCNMFE_ROOT is empty" in line and tree(srv) == before, f"empty root: off ({line})")
    nt = make_session(six, "AVG5x-TSeries-090626-pnb99-331um-3z-000", tif=None)
    line = P.stage_tif_forcnmfe(nt, False, mapping=m)
    check("no session .tif" in line and tree(srv) == before, f"no local tif: nothing done ({line})")
    line = P.stage_tif_forcnmfe(sd, False, root=str(tmp / "not_there"), mapping=m)
    check("not reachable" in line and not (tmp / "not_there").exists(),
          f"unreachable root: not copied, nothing created ({line})")

    print("2. first copy into the mapped folder, then idempotent")
    line = P.stage_tif_forcnmfe(sd, True, mapping=m)
    check("would copy" in line and "6odorDualDiffRew" in line and tree(srv) == before,
          f"dry run: says where, writes nothing ({line})")
    line = P.stage_tif_forcnmfe(sd, False, mapping=m)
    dest = srv / "6odorDualDiffRew" / f"{sd.name}.tif"
    check("copied" in line and "identical" in line and dest.read_bytes() == (sd / f"{sd.name}.tif").read_bytes(),
          f"copied and read back ({line})")
    after = tree(srv)
    check(set(after) - set(before) == {str(dest.relative_to(srv))}, "exactly one new file on the server, nothing else")
    line = P.stage_tif_forcnmfe(sd, False, mapping=m)
    check("already in for CNMFe" in line and tree(srv) == after, f"second push: already there, nothing written ({line})")

    print("3. a copy elsewhere in the tree counts; a dot-folder copy does not")
    sd2 = make_session(six, "AVG5x-TSeries-090726-pnb99-332um-3z-000", tif=b"TWO-" * 900)
    (srv / "misc").mkdir()
    (srv / "misc" / f"{sd2.name}.TIF").write_bytes(b"TWO-" * 900)
    before = tree(srv)
    line = P.stage_tif_forcnmfe(sd2, False, mapping=m)
    check("already in for CNMFe" in line and "misc" in line and tree(srv) == before,
          f"identical copy in another folder (any case): no second copy ({line})")
    sd3 = make_session(six, "AVG5x-TSeries-090826-pnb99-333um-3z-000", tif=b"THR-" * 900)
    (srv / ".parked").mkdir()
    (srv / ".parked" / f"{sd3.name}.tif").write_bytes(b"THR-" * 900)
    line = P.stage_tif_forcnmfe(sd3, False, mapping=m)
    check("copied" in line and (srv / "6odorDualDiffRew" / f"{sd3.name}.tif").exists(),
          f"dot-folder copy ignored, real copy made ({line})")

    print("4. a DIFFERENT copy is reported and never overwritten")
    sd4 = make_session(six, "AVG5x-TSeries-090926-pnb99-334um-3z-000", tif=b"NEW-" * 1000)
    old = srv / "6odorDualDiffRew" / f"{sd4.name}.tif"
    old.write_bytes(b"OLD-" * 1000)                          # same size, other movie
    before = tree(srv)
    line = P.stage_tif_forcnmfe(sd4, False, mapping=m)
    check("WARNING" in line and "DIFFERENT" in line and "NOT overwritten" in line
          and old.read_bytes() == b"OLD-" * 1000 and tree(srv) == before,
          f"same size, other contents: warned, untouched ({line})")
    old.write_bytes(b"OLD-" * 10)                            # an interrupted copy
    line = P.stage_tif_forcnmfe(sd4, False, mapping=m)
    check("DIFFERENT" in line and old.read_bytes() == b"OLD-" * 10, f"truncated copy: warned, untouched ({line})")

    print("5. per-animal folders; unmapped or missing folders are never created")
    a99 = make_session(odor3, "AVG5x-TSeries-092926-pnb99-518um-3z-000")
    a98 = make_session(odor3, "AVG5x-TSeries-071626-pnb98-442um-3z-000")
    ax = make_session(odor3, "AVG5x-TSeries-092926-pnb200-500um-3z-000")
    line = P.stage_tif_forcnmfe(a99, False, mapping=m)
    check((srv / "bla pnb 3odor" / "cohort5" / f"{a99.name}.tif").exists(), f"pnb99 3odor -> cohort5 ({line})")
    line = P.stage_tif_forcnmfe(a98, False, mapping=m)
    check((srv / "bla pnb 3odor" / "cohort4" / f"{a98.name}.tif").exists(), f"pnb98 3odor -> cohort4 ({line})")
    before = tree(srv)
    line = P.stage_tif_forcnmfe(ax, False, mapping=m)
    check("no for-CNMFe folder mapped" in line and "pnb200" in line and tree(srv) == before,
          f"animal without an entry and no '*': not copied ({line})")
    cta = make_session(data / "vCA1" / "CTA", "AVG5x-TSeries-080626-pnb98-480um-3z-000")
    line = P.stage_tif_forcnmfe(cta, False, mapping=m)
    check("does not exist" in line and not (srv / "pnb CTA").exists() and tree(srv) == before,
          f"mapped folder missing: not copied, not created ({line})")
    bla = make_session(data / "BLA" / "Valence", "AVG5x-TSeries-010126-bla50-500um-3z-000")
    line = P.stage_tif_forcnmfe(bla, False, mapping=m)
    check("no for-CNMFe folder mapped for BLA/Valence" in line and tree(srv) == before,
          f"area/task not in the map: not copied ({line})")
    outside = make_session(tmp / "elsewhere", "AVG5x-TSeries-010226-pnb99-500um-3z-000")
    line = P.stage_tif_forcnmfe(outside, False, mapping=m)
    check("no for-CNMFe folder mapped" in line and tree(srv) == before,
          f"session outside DATA_PARENT: not copied ({line})")
    tf = make_session(six, "AVG5x-TSeries-091026-pnb99-335um-3z-000", ext=".tiff")
    line = P.stage_tif_forcnmfe(tf, False, mapping=m)
    check((srv / "6odorDualDiffRew" / f"{tf.name}.tiff").exists(), f".tiff keeps its name ({line})")

    print("6. stage_one: bundle as before, tif goes to for CNMFe, not into the outbox")
    s1 = make_session(six, "AVG5x-TSeries-082626-pnb99-399um-3z-000")
    bundle = exchange / "outbox" / "Taylor" / "vCA1" / "6odorDualDiffRew" / s1.name
    P_map, P.FORCNMFE_MAP = P.FORCNMFE_MAP, tmp / "map.json"
    import json
    P.FORCNMFE_MAP.write_text(json.dumps(m), encoding="utf-8")
    _, out = quiet(P.stage_one, s1, str(exchange), True, "Taylor")
    check("tif: would copy" in out and not bundle.exists()
          and not (srv / "6odorDualDiffRew" / f"{s1.name}.tif").exists(),
          "stage_one dry run: tif line, nothing written")
    _, out = quiet(P.stage_one, s1, str(exchange), False, "Taylor")
    check("tif: copied to for CNMFe/6odorDualDiffRew" in out
          and (srv / "6odorDualDiffRew" / f"{s1.name}.tif").exists(), "stage_one copies the tif")
    check((bundle / "review_neuron.mat").exists() and (bundle / f"{s1.name}.mat").exists()
          and (bundle / "run_final_review.m").exists() and not (bundle / f"{s1.name}.tif").exists(),
          "bundle unchanged: movie .mat in it, no tif")
    P.FORCNMFE_MAP = P_map

    print("7. the real config/forcnmfe_folders.json")
    real = P.load_forcnmfe_map()
    check(real and not any(k.startswith("_") for k in real), f"loads, notes dropped ({len(real)} area/task keys)")
    for sdx, want in ((a99, "bla pnb 3odor/cohort5"), (a98, "bla pnb 3odor/cohort4"),
                      (sd, "6odorDualDiffRew"), (cta, "pnb CTA"), (ax, None), (bla, None)):
        got = P.forcnmfe_folder(sdx, real)[1]
        check(got == want, f"{P.session_rel(sdx).as_posix()} -> {got}")
    check(P.load_forcnmfe_map(tmp / "missing.json") == {}, "missing map file -> {} (nothing copied)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
