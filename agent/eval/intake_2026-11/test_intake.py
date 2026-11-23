"""Temp-dir test for intake_legacy_sessions.py.  Never touches the server or
the real data tree: both roots are temp folders."""
import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("intake", HERE / "intake_legacy_sessions.py")
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def write(p: Path, data: bytes, t=1_700_000_000):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    os.utime(p, (t, t))


def make_src(root: Path, name: str, cells: int, movie_mat=True, extra_movie=False):
    d = root / "srv" / name
    write(d / "neuron.mat", b"N" * 100)
    write(d / "C_raw.txt", b"".join(b"1 2 3\n" for _ in range(cells)))
    write(d / "A.txt", b"A")
    write(d / "ROIs.jpg", b"J")
    write(d / "._A.txt", b"mac junk")
    write(d / f"{name}.tif", os.urandom(3 * (1 << 20)))
    if movie_mat:
        write(d / f"{name}.mat", b"M" * 1000)
    if extra_movie:
        write(d / "other.tif", b"T")
    write(d / f"{name}_neurons" / "Neuron_1.png", b"P")
    return d


def snapshot(root: Path):
    return {str(p): (p.stat().st_size, p.stat().st_mtime) for p in root.rglob("*") if p.is_file()}


with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    src_root, dst_root, out_dir = t / "X", t / "D", t / "out"
    (dst_root / "BLA" / "Valence").mkdir(parents=True)
    (dst_root / "BLA" / ".excluded" / "Valence" / "TAKEN-NAME").mkdir(parents=True)
    make_src(src_root, "OLD-blay13", 3)
    make_src(src_root, "Julian_movie", 19)
    make_src(src_root, "WRONGCOUNT", 5)
    make_src(src_root, "TWOMOVIES", 2, extra_movie=True)
    make_src(src_root, "COLLIDE", 2)
    live = dst_root / "BLA" / "Valence" / "LIVE"
    write(live / "neuron.mat", b"L")
    write(live / "C_raw.txt", b"1\n" * 20)
    before_src = snapshot(src_root)

    good = [("BLA", "Valence", "NEW-bla13", "srv/OLD-blay13", 3, "full"),
            ("BLA", "Valence", "LIVE", "srv/Julian_movie", 20, "tif_only")]

    print("dry run")
    rc = intake.run(good, src_root, dst_root, True, out_dir)
    check(rc == 0, "dry run of a valid plan returns 0")
    check(not (dst_root / "BLA" / "Valence" / "NEW-bla13").exists(), "dry run creates nothing")
    check(not out_dir.exists(), "dry run writes no sessions file")

    print("refusals")
    for bad, why in [
        (("BLA", "Valence", "X1", "srv/WRONGCOUNT", 9, "full"), "cell count mismatch"),
        (("BLA", "Valence", "X2", "srv/TWOMOVIES", 2, "full"), "two movies"),
        (("BLA", "Valence", "TAKEN-NAME", "srv/COLLIDE", 2, "full"), "name taken in a dot folder"),
        (("BLA", "Valence", "LIVE", "srv/COLLIDE", 2, "full"), "destination exists"),
        (("BLA", "Valence", "X3", "srv/MISSING", 2, "full"), "source missing"),
        (("BLA", "NoTask", "X4", "srv/COLLIDE", 2, "full"), "task folder missing"),
        (("BLA", "Valence", "LIVE", "srv/COLLIDE", 99, "tif_only"), "tif_only count mismatch"),
    ]:
        rc = intake.run(good + [bad], src_root, dst_root, False, out_dir)
        check(rc == 1, f"refused: {why}")
        check(not (dst_root / "BLA" / "Valence" / "NEW-bla13").exists(),
              f"nothing copied when one entry is bad ({why})")

    print("real run")
    rc = intake.run(good, src_root, dst_root, False, out_dir)
    check(rc == 0, "valid plan copies")
    nd = dst_root / "BLA" / "Valence" / "NEW-bla13"
    check((nd / "NEW-bla13.tif").exists(), "movie renamed to the destination name")
    check(not (nd / "OLD-blay13.tif").exists(), "old movie name not present")
    check(not (nd / "OLD-blay13.mat").exists() and not (nd / "NEW-bla13.mat").exists(),
          "converted movie .mat not copied")
    check(not (nd / "._A.txt").exists(), "mac resource-fork files skipped")
    check((nd / "NEW-bla13_neurons" / "Neuron_1.png").exists(), "_neurons folder renamed")
    check(intake.c_raw_rows(nd / "C_raw.txt") == 3, "curation copied")
    check(hashlib.md5((nd / "NEW-bla13.tif").read_bytes()).hexdigest() ==
          hashlib.md5((src_root / "srv" / "OLD-blay13" / "OLD-blay13.tif").read_bytes()).hexdigest(),
          "movie bytes identical")
    check(not list(dst_root.rglob("*.partial")), "no .partial files left")
    man = (nd / "INTAKE_MANIFEST.txt").read_text()
    check("renamed_from: OLD-blay13" in man and "NEW-bla13.tif" in man, "manifest records the rename")
    check(all(ord(c) < 128 for c in man), "manifest is ASCII")
    check((live / "LIVE.tif").exists() and (live / "INTAKE_MANIFEST.txt").exists(),
          "tif_only adds the movie and a manifest")
    check(live.joinpath("neuron.mat").read_bytes() == b"L", "tif_only leaves the live curation alone")
    check(not (live / "A.txt").exists(), "tif_only copies no curation files")
    sf = (out_dir / "bootstrap_sessions_BLA.txt").read_text()
    check(str(nd) in sf and str(live) in sf, "sessions file lists both")
    check(snapshot(src_root) == before_src, "source tree unchanged (sizes and mtimes)")

    print("second run refuses")
    rc = intake.run(good, src_root, dst_root, False, out_dir)
    check(rc == 1, "re-running the same plan is refused (no overwrite)")

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
