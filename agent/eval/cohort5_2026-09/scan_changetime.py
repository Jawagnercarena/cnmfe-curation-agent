"""Read-only scan: list every file/dir under ROOT whose NTFS ChangeTime falls in
[T0, T1].  ChangeTime is set by the filesystem on any data or metadata write and
cannot be preserved by shutil.copy2, so it identifies what a copy actually wrote
even though the copied files carry the source's mtime.

usage: scan_changetime.py ROOT "YYYY-mm-dd HH:MM" "YYYY-mm-dd HH:MM" OUTFILE
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

k32 = ctypes.windll.kernel32
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p,
                            wt.DWORD, wt.DWORD, wt.HANDLE]
k32.GetFileInformationByHandleEx.restype = wt.BOOL
k32.GetFileInformationByHandleEx.argtypes = [wt.HANDLE, wt.DWORD, ctypes.c_void_p, wt.DWORD]
k32.CloseHandle.argtypes = [wt.HANDLE]
INVALID = wt.HANDLE(-1).value
FILE_SHARE_ALL = 1 | 2 | 4
OPEN_EXISTING = 3
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FileBasicInfo = 0


class FILE_BASIC_INFO(ctypes.Structure):
    _fields_ = [("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("FileAttributes", wt.DWORD)]


EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


def ft_to_local(ft):
    return (EPOCH + timedelta(microseconds=ft / 10)).astimezone().replace(tzinfo=None)


def basic_info(path):
    h = k32.CreateFileW("\\\\?\\" + path, 0, FILE_SHARE_ALL, None, OPEN_EXISTING,
                        FILE_FLAG_BACKUP_SEMANTICS, None)
    if h == INVALID or h is None:
        return None
    try:
        fbi = FILE_BASIC_INFO()
        if not k32.GetFileInformationByHandleEx(h, FileBasicInfo, ctypes.byref(fbi),
                                                ctypes.sizeof(fbi)):
            return None
        return fbi
    finally:
        k32.CloseHandle(h)


def main():
    root, t0, t1, outfile = sys.argv[1:5]
    t0 = datetime.strptime(t0, "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(t1, "%Y-%m-%d %H:%M")
    hits = []
    n_seen = 0
    n_fail = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            n_seen += 1
            fbi = basic_info(p)
            if fbi is None:
                n_fail += 1
                continue
            ch = ft_to_local(fbi.ChangeTime)
            if t0 <= ch <= t1:
                is_dir = name in dirnames
                size = 0 if is_dir else os.path.getsize(p)
                hits.append((ch, ft_to_local(fbi.CreationTime), ft_to_local(fbi.LastWriteTime),
                             size, is_dir, os.path.relpath(p, root)))
    hits.sort(key=lambda r: (r[5].lower()))
    by_session = defaultdict(list)
    for r in hits:
        parts = r[5].split(os.sep)
        key = os.sep.join(parts[:3]) if len(parts) >= 3 else parts[0]
        by_session[key].append(r)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"root={root} window={t0}..{t1} scanned={n_seen} unreadable={n_fail} hits={len(hits)}\n")
        for key in sorted(by_session, key=str.lower):
            rows = by_session[key]
            f.write(f"\n== {key}  ({len(rows)} entries)\n")
            f.write("   change_time          created              mtime                size        path\n")
            for ch, cr, mt, size, is_dir, rel in rows:
                tag = "<DIR>" if is_dir else f"{size:>11d}"
                f.write(f"   {ch:%Y-%m-%d %H:%M:%S}  {cr:%Y-%m-%d %H:%M:%S}  {mt:%Y-%m-%d %H:%M:%S}  {tag}  {rel}\n")
    print(f"scanned={n_seen} unreadable={n_fail} hits={len(hits)} sessions={len(by_session)} -> {outfile}")
    for key in sorted(by_session, key=str.lower):
        rows = by_session[key]
        new = sum(1 for r in rows if not r[4] and t0 <= r[1] <= t1)
        over = sum(1 for r in rows if not r[4] and not (t0 <= r[1] <= t1))
        print(f"  {len(rows):3d} entries  {new:3d} created  {over:3d} overwritten   {key}")


if __name__ == "__main__":
    main()
