"""Re-make the 12 cohort-5 valence AVG5x movies with the installed motion-correction rule.

    <suite2p env python> remake_valence.py --server-dir "X:/Jeremy/2p data/valence"
        --delivered-dir "D:/2p data/valence" --scripts-dir "D:/2p data/scripts"
        --scratch <local scratch folder> --log-dir <folder>
        [--only SUBSTR ...] [--slots 2] [--test-copy N] [--go]

Run it with the suite2p env's python: the process_2p.py children are started with
the same interpreter (sys.executable), so no interpreter path is written here.

Per session, in its own task folder <scratch>/<day>-<animal>/ (so no process_2p.py
run ever lists a folder that another run is writing or deleting):
  1. COPY    the raw TSeries folder (every file, References/ included) from the
             server into <task>/_incoming/<TSeries>, hashing each file (SHA-256)
             while it is read. VERIFY: same file set and byte sizes as the server
             listing; the local file re-hashed AND the server file hashed a second
             time (an independent read) both equal the hash taken during the copy.
             Only then is the folder renamed (same volume) to <task>/<TSeries>.
             The XML frame count is read from the verified copy.
  2. REGISTER  process_2p.py --task-dir <task> --only <TSeries> --skip-average
             (at most --slots sessions at once).
  3. AVERAGE   process_2p.py --task-dir <task> --only <TSeries> --skip-suite2p
             (one at a time: never two Fiji steps at once). Writes
             <task>/AVG5x-<TSeries>.tif, <task>/mc_record/, <task>/mc_qc.csv and,
             when the movie verifies and the QC row is unflagged, removes the
             scratch TSeries copy (process_2p.py's own rule). A flagged session
             keeps it.
  4. CHECK   new movie frames == sum(XML frames) // 5 == delivered movie frames;
             QC row read back.
Results in --log-dir: driver.log, reg_<key>.log / avg_<key>.log per session,
copy_manifest.csv (every file, bytes, SHA-256), summary.csv.

Read-only on the server: files there are only opened for reading. Writes go to
--scratch and --log-dir, both refused if on X: or a UNC path. This script deletes
nothing; the only removal is process_2p.py's deletion of a verified scratch copy.
The delivered movies are only read (frame count). Swapping the new movies in is a
separate step (swap_in_remade_movies.py). Dry run by default; --go acts.
"""
import argparse
import csv
import hashlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SESSIONS = [
    "TSeries-091526-pnb99-442um-3z-000",
    "TSeries-091526-pnb102-486um-3z-000",
    "TSeries-091526-pnb104-484um-3z-000",
    "TSeries-091526-pnb105-403um-35z-000",
    "TSeries-091526-pnb106-235um-4z-000",
    "TSeries-091526-pnb109-375um-25z-000",
    "TSeries-091626-pnb99-441um-3z-000",
    "TSeries-091626-pnb102-464um-3z-000",
    "TSeries-091626-pnb104-510um-3z-000",
    "TSeries-091626-pnb105-410um-35z-000",
    "TSeries-091626-pnb106-242um-4z-000",
    "TSeries-091626-pnb109-362um-25z-000",
]
FACTOR = 5
BLOCK = 16 << 20
# Free bytes the scratch drive must keep after a copy: two registrations in flight
# each grow by about 3 x the raw size (data_raw.bin, data.bin, reg_tif).
FREE_MARGIN = 400 << 30
MAX_PATH_CHARS = 259

_lock = threading.Lock()
_log_fh = None
RESULTS = {}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with _lock:
        print(line, flush=True)
        if _log_fh is not None:
            _log_fh.write(line + "\n")
            _log_fh.flush()


def note(name, **kw):
    with _lock:
        RESULTS.setdefault(name, {}).update(kw)


def refuse_server(p, what):
    s = str(p)
    if s.startswith("\\\\") or s.startswith("//") or s[:2].upper() == "X:":
        sys.exit(f"refusing {what} on the server: {p}")


def key_of(name):
    parts = name.split("-")          # TSeries, 091526, pnb99, 442um, ...
    return f"{parts[1]}-{parts[2]}"


def day_of(name):
    return name.split("-")[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def server_listing(src):
    return {f.relative_to(src).as_posix(): f.stat().st_size
            for f in sorted(src.rglob("*")) if f.is_file()}


def xml_frames(folder):
    xmls = sorted(folder.glob("*.xml"))
    if len(xmls) != 1:
        raise RuntimeError(f"expected one .xml in {folder}, found {len(xmls)}")
    text = xmls[0].read_text(encoding="utf-8-sig", errors="replace")
    starts = [i for i in _find_all(text, "<Sequence ")]
    bounds = starts + [len(text)]
    return [text.count("<Frame ", bounds[i], bounds[i + 1]) for i in range(len(starts))]


def _find_all(text, token):
    i = text.find(token)
    while i >= 0:
        yield i
        i = text.find(token, i + 1)


def tif_frames(path):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        meta = tif.imagej_metadata or {}
        if "images" in meta:
            return int(meta["images"])
        return len(tif.pages)


def qc_row(csv_path, session):
    if not csv_path.is_file():
        return None
    with open(csv_path, newline="", encoding="ascii", errors="replace") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("session") == session]
    return rows[-1] if rows else None


# --------------------------------------------------------------------------- #
# copy + verify
# --------------------------------------------------------------------------- #

def copy_and_verify(src, dst, manifest, limit=None):
    """Copy src -> dst (dst must not exist) and verify. Returns (ok, detail, nfiles, nbytes)."""
    listing = server_listing(src)
    rels = list(listing)[:limit] if limit else list(listing)
    if dst.exists():
        return False, f"{dst} already exists", 0, 0
    hashes = {}
    total = 0
    for rel in rels:
        s, d = src / rel, dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with open(s, "rb") as fi, open(d, "xb") as fo:
            for block in iter(lambda: fi.read(BLOCK), b""):
                h.update(block)
                fo.write(block)
        shutil.copystat(s, d)
        size = d.stat().st_size
        if size != listing[rel]:
            return False, f"{rel}: copied {size} bytes, server lists {listing[rel]}", 0, 0
        hashes[rel] = h.hexdigest()
        total += size
    local = {f.relative_to(dst).as_posix() for f in dst.rglob("*") if f.is_file()}
    if local != set(rels):
        return False, f"file sets differ: extra {sorted(local - set(rels))[:3]} missing {sorted(set(rels) - local)[:3]}", 0, 0
    for rel in rels:
        h_local = sha256_file(dst / rel)
        h_server = sha256_file(src / rel)
        if not (h_local == h_server == hashes[rel]):
            return False, (f"{rel}: hash mismatch copy {hashes[rel][:12]} local {h_local[:12]} "
                           f"server-reread {h_server[:12]}"), 0, 0
    with _lock:
        new = not manifest.exists()
        with open(manifest, "a", newline="", encoding="ascii") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["session", "file", "bytes", "sha256"])
            for rel in rels:
                w.writerow([src.name, rel, listing[rel], hashes[rel]])
    return True, "verified", len(rels), total


# --------------------------------------------------------------------------- #
# children
# --------------------------------------------------------------------------- #

def run_child(args, log_path, cwd):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("CMD " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n")
        fh.flush()
        p = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        fh.write(f"exit {p.returncode}\n")
    return p.returncode


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    global _log_fh
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", type=Path, required=True)
    ap.add_argument("--delivered-dir", type=Path, required=True)
    ap.add_argument("--scripts-dir", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--test-copy", type=int, default=0,
                    help="copy + verify only the first N files of the first selected "
                         "session into <scratch>/_copytest, then stop")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    for p, what in ((a.scratch, "--scratch"), (a.log_dir, "--log-dir")):
        refuse_server(p.resolve(), what)
    process_2p = a.scripts_dir / "process_2p.py"
    if not process_2p.is_file():
        sys.exit(f"no {process_2p}")
    names = [n for n in SESSIONS if not a.only or any(o.lower() in n.lower() for o in a.only)]
    if not names:
        sys.exit("no session selected")

    if a.test_copy:
        name = names[0]
        src = a.server_dir / day_of(name) / name
        dst = a.scratch / "_copytest" / name
        a.log_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ok, detail, n, b = copy_and_verify(src, dst, a.log_dir / "copytest_manifest.csv",
                                           limit=a.test_copy)
        print(f"test copy {name}: {'OK' if ok else 'FAILED'} {detail}; {n} files, {b} bytes, "
              f"{time.time() - t0:.0f} s -> {dst}")
        return 0 if ok else 1

    # ---- plan ----
    free = shutil.disk_usage(a.scratch.resolve().anchor).free
    print(f"{'GO' if a.go else 'DRY RUN'}: {len(names)} session(s); interpreter {sys.executable}")
    print(f"  server {a.server_dir} (read only); delivered {a.delivered_dir} (read only)")
    print(f"  scratch {a.scratch} (free {free / 2**30:.0f} GiB); logs {a.log_dir}; slots {a.slots}")
    plan, refused = [], []
    for name in names:
        src = a.server_dir / day_of(name) / name
        task = a.scratch / key_of(name)
        delivered = a.delivered_dir / f"AVG{FACTOR}x-{name}.tif"
        why = None
        if not src.is_dir():
            why = f"no server folder {src}"
        elif not delivered.is_file():
            why = f"no delivered movie {delivered}"
        elif task.exists():
            why = f"task folder already exists: {task}"
        if why:
            refused.append((name, why))
            continue
        listing = server_listing(src)
        n_ome = sum(1 for r in listing if r.endswith(".ome.tif") and "/" not in r)
        # Windows MAX_PATH: the copy goes through <task>/_incoming/<TSeries>/<rel>.
        # 2026-11-24: four sessions (25z / 35z names) hit 260 chars under a long
        # --scratch and failed at the first References file.
        longest = max(len(str(task.resolve() / "_incoming" / name / r)) for r in listing)
        if longest > MAX_PATH_CHARS:
            refused.append((name, f"scratch path would be {longest} chars (limit "
                                  f"{MAX_PATH_CHARS}); use a shorter --scratch"))
            continue
        plan.append((name, src, task, delivered))
        print(f"  {name}: {len(listing)} files ({n_ome} ome.tif), "
              f"{sum(listing.values()) / 2**30:.1f} GiB -> {task}")
    for name, why in refused:
        print(f"  REFUSED {name}: {why}")
    if refused:
        sys.exit(f"{len(refused)} refused; nothing done.")
    if not a.go:
        return 0

    a.log_dir.mkdir(parents=True, exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    _log_fh = open(a.log_dir / "driver.log", "a", encoding="ascii", errors="replace")
    manifest = a.log_dir / "copy_manifest.csv"
    py = sys.executable
    log(f"START {len(plan)} sessions; interpreter {py}; process_2p {process_2p}")

    reg_q = queue.Queue(maxsize=1)
    avg_q = queue.Queue()

    def copier():
        try:
            for name, src, task, delivered in plan:
                note(name, key=key_of(name), status="copying")
                need = sum(server_listing(src).values())
                while shutil.disk_usage(str(a.scratch)).free < need + FREE_MARGIN:
                    log(f"COPY {key_of(name)} waits for disk space")
                    time.sleep(120)
                incoming = task / "_incoming" / name
                t0 = time.time()
                log(f"COPY {key_of(name)} start")
                try:
                    ok, detail, n, b = copy_and_verify(src, incoming, manifest)
                except Exception as exc:          # noqa: BLE001
                    ok, detail, n, b = False, f"{type(exc).__name__}: {exc}", 0, 0
                if not ok:
                    log(f"COPY {key_of(name)} FAILED: {detail} (left in {incoming})")
                    note(name, status=f"copy failed: {detail}", t_copy_s=round(time.time() - t0))
                    continue
                final = task / name
                os.rename(incoming, final)
                frames = xml_frames(final)
                note(name, server_files=n, server_bytes=b, copy_verified="yes",
                     xml_cycles=len(frames), xml_raw_frames=sum(frames),
                     expected_avg_frames=sum(frames) // FACTOR, t_copy_s=round(time.time() - t0))
                log(f"COPY {key_of(name)} verified: {n} files, {b / 2**30:.1f} GiB, "
                    f"XML {len(frames)} cycles / {sum(frames)} frames, {time.time() - t0:.0f} s")
                reg_q.put((name, task, delivered))
        except Exception as exc:                  # noqa: BLE001
            log(f"COPIER CRASHED: {type(exc).__name__}: {exc}")
        finally:
            for _ in range(a.slots):
                reg_q.put(None)

    def registrar(slot):
        while True:
            item = reg_q.get()
            if item is None:
                return
            name, task, delivered = item
            k = key_of(name)
            note(name, status="registering")
            log(f"REG {k} start (slot {slot})")
            t0 = time.time()
            try:
                rc = run_child([py, str(process_2p), "--task-dir", str(task), "--only", name,
                                "--skip-average"], a.log_dir / f"reg_{k}.log", str(a.scripts_dir))
            except Exception as exc:              # noqa: BLE001
                rc = f"{type(exc).__name__}: {exc}"
            reg = task / name / "suite2p" / "plane0" / "reg_tif"
            made = len(list(reg.glob("*.tif"))) if reg.is_dir() else 0
            note(name, t_reg_s=round(time.time() - t0))
            if rc != 0 or not made:
                log(f"REG {k} FAILED: exit {rc}, {made} reg_tif files")
                note(name, status=f"registration failed (exit {rc})")
                continue
            log(f"REG {k} done: {made} reg_tif files, {time.time() - t0:.0f} s")
            avg_q.put(item)

    def averager():
        while True:
            item = avg_q.get()
            if item is None:
                return
            name, task, delivered = item
            k = key_of(name)
            note(name, status="averaging")
            log(f"AVG {k} start")
            t0 = time.time()
            try:
                rc = run_child([py, str(process_2p), "--task-dir", str(task), "--only", name,
                                "--skip-suite2p"], a.log_dir / f"avg_{k}.log", str(a.scripts_dir))
            except Exception as exc:              # noqa: BLE001
                rc = f"{type(exc).__name__}: {exc}"
            note(name, t_avg_s=round(time.time() - t0))
            new_tif = task / f"AVG{FACTOR}x-{name}.tif"
            row = qc_row(task / "mc_qc.csv", name)
            kept = (task / name).exists()
            info = dict(tseries_kept="yes" if kept else "no")
            try:
                info["new_frames"] = tif_frames(new_tif) if new_tif.is_file() else ""
                info["delivered_frames"] = tif_frames(delivered)
            except Exception as exc:              # noqa: BLE001
                info["new_frames"] = f"read error {type(exc).__name__}"
            if row:
                for f in ("bad_pct", "degraded_pct", "at_limit_pct", "ref_offset_y",
                          "ref_offset_x", "flag"):
                    info[f] = row.get(f, "")
            note(name, **info)
            exp = RESULTS[name].get("expected_avg_frames")
            frames_ok = (info.get("new_frames") == exp == info.get("delivered_frames"))
            note(name, frames_ok="yes" if frames_ok else "NO")
            if rc != 0 or not new_tif.is_file():
                status = f"averaging failed (exit {rc})"
            elif row is None:
                status = "no QC row"
            elif row.get("flag"):
                status = f"QC FLAGGED: {row['flag']}"
            elif not frames_ok:
                status = "frame count mismatch"
            else:
                status = "ok"
            note(name, status=status)
            log(f"AVG {k} {status}: new {info.get('new_frames')} / expected {exp} / delivered "
                f"{info.get('delivered_frames')} frames; bad {info.get('bad_pct')}% degraded "
                f"{info.get('degraded_pct')}% at limit {info.get('at_limit_pct')}%; "
                f"TSeries kept {info['tseries_kept']}; {time.time() - t0:.0f} s")

    threads = [threading.Thread(target=copier, name="copier")]
    threads += [threading.Thread(target=registrar, args=(i + 1,), name=f"reg{i + 1}")
                for i in range(a.slots)]
    avg_thread = threading.Thread(target=averager, name="avg")
    for t in threads + [avg_thread]:
        t.start()
    for t in threads:
        t.join()
    avg_q.put(None)
    avg_thread.join()

    cols = ["key", "session", "status", "server_files", "server_bytes", "copy_verified",
            "xml_cycles", "xml_raw_frames", "expected_avg_frames", "new_frames",
            "delivered_frames", "frames_ok", "bad_pct", "degraded_pct", "at_limit_pct",
            "ref_offset_y", "ref_offset_x", "flag", "tseries_kept", "t_copy_s", "t_reg_s",
            "t_avg_s"]
    with open(a.log_dir / "summary.csv", "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for name, *_ in plan:
            w.writerow(dict(RESULTS.get(name, {}), session=name, key=key_of(name)))
    n_ok = sum(1 for name, *_ in plan if RESULTS.get(name, {}).get("status") == "ok")
    log(f"ALL DONE: {n_ok} of {len(plan)} ok; summary {a.log_dir / 'summary.csv'}")
    return 0 if n_ok == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
