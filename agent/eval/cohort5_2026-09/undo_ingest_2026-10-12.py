"""Undo the 2026-10-12 08:45 ingest of four cohort-5 pilot sessions.

What that ingest did (measured with NTFS ChangeTime, see the LOG entry for
2026-10-12): it wrote files into exactly four local sessions under
vCA1/6odorDualDiffRew (082626 for pnb102, pnb105, pnb106, pnb99), rewrote the
labels_provenance.txt timestamp of 108 previously ingested sessions, and the
vCA1 watcher then retrained classifier.joblib on 163 sessions (161 + pnb102 +
pnb99; pnb105/pnb106 were skipped as a size mismatch because their returns
were reviews of the pre-2026-10-06 candidate set).

This script, per session:
  1. DELETES the files the ingest created (labels.mat, ROIs.jpg,
     merge_log_*.mat, labels_provenance.txt).  Each one must have an NTFS
     creation date of 2026-10-12, or the script refuses to touch it.
  2. RESTORES overwritten files that still exist elsewhere:
       spatial_footprints.mat, review_summary.txt  <- <session>/.rerun_mincorr070_2026-10-06/
       review_neuron.mat, review_report.pdf, Ybg_weights.mat, Cn.mat, pnr.mat
                                                   <- server outbox/Taylor/... (READ ONLY)
     A file is copied only when the local copy differs (size or mtime) from
     the 2026-10-06 source; unchanged files are reported and left alone.
  3. REPORTS the overwritten files for which no pre-ingest copy exists anywhere
     (neuron.mat, Coor.mat, A/C/C_df/C_raw/S.txt, the re-rendered kept-cell
     PNGs).  They are left in place; only a headless re-run regenerates them.
Then it rolls the vCA1 model back to classifier_prepilot_2026-09-24.joblib
(the 161-session model that was deployed before the ingest), after saving a
copy of today's 163-session model to --model-backup.

The server is never written to: the only server access is shutil.copy2 FROM
the outbox TO the local session.  Every delete is os.remove on a path that is
verified to sit under DATA_PARENT.  There is no rmtree, move or rename here.

usage:
  python undo_ingest_2026-10-12.py --dry-run
  python undo_ingest_2026-10-12.py --go --model-backup <dir>
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import hashlib
import os
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))          # agent/
from local_config import DATA_PARENT, EXCHANGE_ROOT, REPO_ROOT  # noqa: E402

INGEST_DAY = date(2026, 10, 12)
SOURCE_DAY = date(2026, 10, 6)                      # the pre-ingest versions
TASK_DIR = DATA_PARENT / "vCA1" / "6odorDualDiffRew"
OUTBOX = Path(EXCHANGE_ROOT) / "outbox" / "Taylor" / "vCA1" / "6odorDualDiffRew"
SNAPSHOT = ".rerun_mincorr070_2026-10-06"
SESSIONS = [
    "AVG5x-TSeries-082626-pnb102-479um-3z-000",
    "AVG5x-TSeries-082626-pnb105-397um-35z-000",
    "AVG5x-TSeries-082626-pnb106-205um-4z-000",
    "AVG5x-TSeries-082626-pnb99-399um-3z-000",
]
CREATED = ["labels.mat", "ROIs.jpg", "labels_provenance.txt"]   # plus merge_log_*.mat
FROM_SNAPSHOT = ["spatial_footprints.mat", "review_summary.txt"]
FROM_OUTBOX = ["review_neuron.mat", "review_report.pdf", "Ybg_weights.mat", "Cn.mat", "pnr.mat"]
UNRECOVERABLE = ["neuron.mat", "Coor.mat", "A.txt", "C.txt", "C_df.txt", "C_raw.txt", "S.txt"]

MODEL_DIR = REPO_ROOT / "agent" / "model" / "vCA1"
MODEL = MODEL_DIR / "classifier.joblib"
PRE_INGEST_MODEL = MODEL_DIR / "classifier_prepilot_2026-09-24.joblib"
PRE_INGEST_MD5 = "affdfaab98ed6341b87cc5cea223dd85"   # recorded in memory 2026-09-24, re-measured 2026-10-12
INGEST_MODEL_MD5 = "12b7e4041f7aadf155a80cac695a324d"  # the 163-session model written 2026-10-12 08:47:09

# ---------------------------------------------------------------- NTFS times
k32 = ctypes.windll.kernel32
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
k32.GetFileInformationByHandleEx.restype = wt.BOOL
k32.GetFileInformationByHandleEx.argtypes = [wt.HANDLE, wt.DWORD, ctypes.c_void_p, wt.DWORD]
k32.CloseHandle.argtypes = [wt.HANDLE]
_INVALID = wt.HANDLE(-1).value
_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


class _FBI(ctypes.Structure):
    _fields_ = [("CreationTime", ctypes.c_longlong), ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong), ("ChangeTime", ctypes.c_longlong),
                ("FileAttributes", wt.DWORD)]


def ntfs_times(path):
    """(creation, change) as naive local datetimes, or None."""
    h = k32.CreateFileW("\\\\?\\" + str(path), 0, 7, None, 3, 0x02000000, None)
    if h == _INVALID or h is None:
        return None
    try:
        fbi = _FBI()
        if not k32.GetFileInformationByHandleEx(h, 0, ctypes.byref(fbi), ctypes.sizeof(fbi)):
            return None
        conv = lambda ft: (_EPOCH + timedelta(microseconds=ft / 10)).astimezone().replace(tzinfo=None)
        return conv(fbi.CreationTime), conv(fbi.ChangeTime)
    finally:
        k32.CloseHandle(h)


# ---------------------------------------------------------------- helpers
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_local(p: Path):
    rp = str(p.resolve()).lower()
    if EXCHANGE_ROOT and rp.startswith(str(Path(EXCHANGE_ROOT)).lower()):
        sys.exit(f"REFUSING: {p} is on the server")
    if not (rp.startswith(str(DATA_PARENT.resolve()).lower())
            or rp.startswith(str(REPO_ROOT.resolve()).lower())):
        sys.exit(f"REFUSING: {p} is outside DATA_PARENT and the repo")


def same(a: Path, b: Path):
    sa, sb = a.stat(), b.stat()
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def mb(n):
    return f"{n / 1e6:.1f} MB"


def label_counts(p: Path):
    try:
        import scipy.io as sio
        d = sio.loadmat(str(p))
        lab = d["labels"].flatten()
        return int((lab == 1).sum()), int((lab == 0).sum())
    except Exception as e:  # noqa: BLE001
        return None


def plan_session(name: str, go: bool):
    sess = TASK_DIR / name
    snap = sess / SNAPSHOT
    out = OUTBOX / name
    print(f"\n=== {name}")
    if not sess.is_dir():
        print("  !! local session missing, skipping"); return
    if not snap.is_dir():
        print(f"  !! {SNAPSHOT} missing, cannot restore spatial_footprints.mat");
    if not out.is_dir():
        print("  !! outbox bundle missing on the server (read-only check), cannot restore review files")

    # 1. delete what the ingest created
    targets = [sess / n for n in CREATED] + sorted(sess.glob("merge_log_*.mat"))
    lp = sess / "labels.mat"
    if lp.exists():
        c = label_counts(lp)
        if c:
            print(f"  ingested labels.mat: {c[0]} keep / {c[1]} delete  (for the record; it is deleted below)")
    for t in targets:
        if not t.exists():
            print(f"  delete   {t.name:28s} -- already absent"); continue
        times = ntfs_times(t)
        if times is None or times[0].date() != INGEST_DAY:
            print(f"  REFUSE   {t.name:28s} -- NTFS creation date is {times[0].date() if times else '?'}, not {INGEST_DAY}")
            continue
        assert_local(t)
        print(f"  delete   {t.name:28s} {mb(t.stat().st_size):>10s}  created {times[0]:%Y-%m-%d %H:%M:%S}")
        if go:
            os.remove(t)

    # 2a. restore from the local 2026-10-06 snapshot
    for n in FROM_SNAPSHOT:
        src, dst = snap / n, sess / n
        if not src.exists():
            print(f"  restore  {n:28s} -- no snapshot copy"); continue
        if dst.exists() and same(src, dst):
            print(f"  keep     {n:28s} -- unchanged (matches snapshot)"); continue
        assert_local(dst)
        print(f"  restore  {n:28s} {mb(src.stat().st_size):>10s}  <- {SNAPSHOT}/  (local now {mb(dst.stat().st_size) if dst.exists() else 'absent'})")
        if go:
            shutil.copy2(str(src), str(dst))

    # 2b. restore from the server outbox (read only on the server)
    for n in FROM_OUTBOX:
        src, dst = out / n, sess / n
        if not src.exists():
            print(f"  restore  {n:28s} -- not in outbox"); continue
        src_day = datetime.fromtimestamp(src.stat().st_mtime).date()
        if src_day != SOURCE_DAY:
            print(f"  REFUSE   {n:28s} -- outbox copy dated {src_day}, expected {SOURCE_DAY}"); continue
        if dst.exists() and same(src, dst):
            print(f"  keep     {n:28s} -- unchanged (matches outbox)"); continue
        assert_local(dst)
        now = f"{mb(dst.stat().st_size)} dated {datetime.fromtimestamp(dst.stat().st_mtime):%Y-%m-%d}" if dst.exists() else "absent"
        print(f"  restore  {n:28s} {mb(src.stat().st_size):>10s}  <- outbox (local now {now})")
        if go:
            shutil.copy2(str(src), str(dst))

    # 3. what cannot be restored
    pngs = [p for p in (sess / f"{name}_neurons").glob("neuron_*.png")
            if (ntfs_times(p) or (None, datetime.min))[1].date() == INGEST_DAY]
    gone = [n for n in UNRECOVERABLE if (sess / n).exists()]
    print(f"  left     {', '.join(gone)} and {len(pngs)} kept-cell PNGs: overwritten by the ingest,"
          f" no pre-ingest copy exists (regenerate only by a headless re-run)")


def rollback_model(go: bool, backup_dir: Path):
    print("\n=== vCA1 classifier")
    cur, pre = md5(MODEL), md5(PRE_INGEST_MODEL)
    print(f"  deployed  {MODEL.name}: md5 {cur}  mtime {datetime.fromtimestamp(MODEL.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
    print(f"  snapshot  {PRE_INGEST_MODEL.name}: md5 {pre}")
    if pre != PRE_INGEST_MD5:
        sys.exit("REFUSING: the 2026-09-24 snapshot does not match its recorded md5")
    if cur == PRE_INGEST_MD5:
        print("  deployed model already equals the snapshot; nothing to do"); return
    if cur != INGEST_MODEL_MD5:
        sys.exit("REFUSING: the deployed model is not the 2026-10-12 08:47 model this script expects")
    # the watcher retrains when any labels.mat is newer than the model: check before rolling back
    snap_mtime = PRE_INGEST_MODEL.stat().st_mtime
    # in a dry run the four ingested labels.mat still exist; they are deleted
    # before this point in a real run, so exclude them from the check either way
    newer = [p for p in (DATA_PARENT / "vCA1").rglob("labels.mat")
             if not any(part.startswith(".") for part in p.relative_to(DATA_PARENT).parts)
             and p.parent.name not in SESSIONS
             and p.stat().st_mtime > snap_mtime]
    if newer:
        print(f"  !! {len(newer)} vCA1 labels.mat newer than the snapshot remain -> the watcher WOULD retrain:")
        for p in newer:
            print(f"     {p.relative_to(DATA_PARENT)}")
        if go:
            sys.exit("REFUSING to roll the model back while newer labels exist")
    else:
        print("  no vCA1 labels.mat newer than the snapshot remain -> the watcher will not retrain")
    bk = backup_dir / "classifier_vCA1_163sessions_ingest_2026-10-12.joblib"
    print(f"  backup    today's model -> {bk}")
    print(f"  rollback  {PRE_INGEST_MODEL.name} -> {MODEL.name} (copy2, keeps the 2026-09-24 mtime)")
    if go:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(MODEL), str(bk))
        assert_local(MODEL)
        shutil.copy2(str(PRE_INGEST_MODEL), str(MODEL))
        print(f"  verify    deployed md5 now {md5(MODEL)}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--go", action="store_true")
    ap.add_argument("--model-backup", type=Path, help="directory for a copy of today's 163-session model (required with --go)")
    args = ap.parse_args()
    if args.go and not args.model_backup:
        sys.exit("--go needs --model-backup <dir>")
    print(f"DATA_PARENT={DATA_PARENT}\nOUTBOX={OUTBOX}\nmode={'GO' if args.go else 'DRY RUN'}")
    for s in SESSIONS:
        plan_session(s, args.go)
    rollback_model(args.go, args.model_backup or Path("<backup-dir>"))
    print("\nDone." if args.go else "\nDry run only; nothing was changed.")


if __name__ == "__main__":
    main()
