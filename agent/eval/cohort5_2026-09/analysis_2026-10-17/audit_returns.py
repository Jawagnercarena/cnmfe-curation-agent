"""READ-ONLY audit: which bundle did Taylor review for each cohort-5 pilot session?
Compares server inbox returns (labels.mat length, review_meta, dates) with the
server outbox bundle and the three local candidate generations.  No writes."""
import os, sys, re, time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent")
import local_config
EX = Path(local_config.EXCHANGE_ROOT)
DP = Path(local_config.DATA_PARENT)
TASK = "vCA1/6odorDualDiffRew"
LOCAL = DP / TASK
NEW = ".rerun_mincorr070_2026-10-06"
OLD = ".prev_est85_2026-10-05"
PILOT = [
 "AVG5x-TSeries-082626-pnb102-479um-3z-000","AVG5x-TSeries-082726-pnb102-497um-3z-000",
 "AVG5x-TSeries-082626-pnb105-397um-35z-000","AVG5x-TSeries-082726-pnb105-391um-35z-000",
 "AVG5x-TSeries-082626-pnb109-398um-26z-000","AVG5x-TSeries-082726-pnb109-405um-25z-000",
 "AVG5x-TSeries-082626-pnb104-404um-3z-000","AVG5x-TSeries-082726-pnb104-416um-3z-000",
 "AVG5x-TSeries-082626-pnb106-205um-4z-000","AVG5x-TSeries-082726-pnb106-201um-4z-000",
 "AVG5x-TSeries-082626-pnb99-399um-3z-000","AVG5x-TSeries-082828-pnb99-436um-3z-000-000".replace("082828","082826"),
]
def ts(p):
    try: return time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    except Exception: return "?"
def ncand(d):
    p = d / "candidate_features.npz"
    if not p.exists(): return None
    z = np.load(str(p), allow_pickle=True)
    return int(np.asarray(z["n_candidates"]).ravel()[0])
def nfoot(d):
    p = d / "spatial_footprints.mat"
    if not p.exists(): return None
    try: return int(loadmat(str(p))["spatial_footprints"].shape[0])
    except Exception as e: return "ERR"
def summ(d):
    p = d / "review_summary.txt"
    if not p.exists(): return "-"
    t = p.read_text(encoding="utf-8", errors="replace")
    m1 = re.search(r"threshold: ([0-9.]+)", t)
    m2 = re.search(r"(\d+) (?:candidates?|neurons?) (?:for|to) review", t)
    m3 = re.search(r"candidates:\s*(\d+)", t)
    return "thr=%s cand=%s rev=%s" % (m1.group(1) if m1 else "?", m3.group(1) if m3 else "?", m2.group(1) if m2 else "?")
def nlab(d):
    p = d / "labels.mat"
    if not p.exists(): return None
    try:
        m = loadmat(str(p)); L = m["labels"]; return (int(L.size), int(np.sum(L == 1)))
    except Exception as e: return ("ERR", str(e)[:40])

print("exchange root set:", bool(local_config.EXCHANGE_ROOT), "| exists:", EX.exists())
inbox = EX / "inbox"; outbox = EX / "outbox"
print("inbox reviewers:", [p.name for p in inbox.iterdir() if p.is_dir()] if inbox.exists() else "NO INBOX")
print()
print("=== LOCAL generations (root = live set; NEW = 0.70 copy; OLD = est85 copy) ===")
for s in PILOT:
    sd = LOCAL / s
    print(s)
    for tag, d in (("root", sd), ("NEW", sd / NEW), ("OLD", sd / OLD)):
        if not d.exists(): print("   %-4s MISSING" % tag); continue
        print("   %-4s ncand=%s nfoot=%s | %s | neuron.mat %s | labels=%s" % (
            tag, ncand(d), nfoot(d), summ(d), ts(d / "neuron.mat") if (d / "neuron.mat").exists() else "-", nlab(d)))
print()
print("=== SERVER outbox bundles for these sessions (any assignee) ===")
if outbox.exists():
    for who in sorted(p for p in outbox.iterdir() if p.is_dir()):
        for s in PILOT:
            b = who / TASK / s
            if b.exists():
                files = sorted(b.iterdir())
                print("%s/%s: %d files | %s | footprints %s (%s) | summary %s" % (
                    who.name, s, len(files), summ(b), nfoot(b), ts(b / "spatial_footprints.mat"), ts(b / "review_summary.txt")))
print()
print("=== SERVER inbox returns for these sessions (any reviewer folder) ===")
if inbox.exists():
    for who in sorted(p for p in inbox.iterdir() if p.is_dir()):
        for s in PILOT:
            cands = [who / TASK / s, who / "6odorDualDiffRew" / s, who / s]
            for r in cands:
                if r.exists():
                    files = sorted(r.iterdir())
                    print("%s/%s  (%s)" % (who.name, s, r.relative_to(inbox).parent))
                    for f in files:
                        print("     %-32s %10s  %s" % (f.name, f.stat().st_size if f.is_file() else "<dir>", ts(f)))
                    print("     labels (n, kept) =", nlab(r), "| return summary:", summ(r))
                    meta = r / "review_meta.txt"
                    if meta.exists():
                        print("     review_meta:", " / ".join(l.strip() for l in meta.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()))
        # any other session folders of this reviewer under the task
        t = who / TASK
        if t.exists():
            extra = [p.name for p in t.iterdir() if p.is_dir() and p.name not in PILOT]
            if extra: print("%s: OTHER session folders under %s: %s" % (who.name, TASK, extra))
