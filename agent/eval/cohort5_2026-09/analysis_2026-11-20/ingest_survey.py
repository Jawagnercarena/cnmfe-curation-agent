"""Read-only survey of the cohort-5 vCA1 sessions: ingest integrity.
Reads D:\\Julian_CNMFe and the server inbox; writes nothing anywhere."""
import sys, re, hashlib, datetime as dt
from pathlib import Path
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import local_config as lc

DATA = Path(lc.DATA_PARENT)
EXCH = Path(lc.EXCHANGE_ROOT) if getattr(lc, "EXCHANGE_ROOT", None) else None
PAT = re.compile(r"pnb(99|102|104|105|106|109)-")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ts(p):
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")


rows = []
for task_dir in sorted((DATA / "vCA1").iterdir()):
    if not task_dir.is_dir() or task_dir.name.startswith("."):
        continue
    for s in sorted(task_dir.iterdir()):
        if not s.is_dir() or s.name.startswith(".") or not PAT.search(s.name):
            continue
        r = {"task": task_dir.name, "name": s.name, "dir": s}
        npz_p = s / "candidate_features.npz"
        if npz_p.exists():
            npz = np.load(str(npz_p), allow_pickle=True)
            n = int(npz["n_candidates"].ravel()[0]) if "n_candidates" in npz.files else int(len(npz["feature_matrix"]))
            auto = npz["auto_rejected"].ravel().astype(int)
            r["N"] = n
            r["auto"] = len(set(auto.tolist()))
            r["npz_t"] = ts(npz_p)
            r["npz_keys"] = npz.files
        lab_p = s / "labels.mat"
        if lab_p.exists():
            d = sio.loadmat(str(lab_p))
            lab = d["labels"].ravel()
            r["L"] = lab.size
            r["keep"] = int((lab == 1).sum())
            r["del"] = int((lab == 0).sum())
            r["other"] = int(((lab != 0) & (lab != 1)).sum())
            r["motion"] = int(d["motion_delete"].ravel().sum()) if "motion_delete" in d else None
            r["lab_t"] = ts(lab_p)
            r["lab_vars"] = [k for k in d if not k.startswith("__")]
        prov = s / "labels_provenance.txt"
        if prov.exists():
            t = prov.read_text(encoding="utf-8", errors="replace")
            m1 = re.search(r"reviewer:\s*(.+)", t)
            m2 = re.search(r"ingested:\s*(.+)", t)
            m3 = re.search(r"source:\s*(.+)", t)
            r["rev"] = m1.group(1).strip() if m1 else "?"
            r["ing"] = m2.group(1).strip()[5:16] if m2 else "?"
            r["src"] = m3.group(1).strip() if m3 else None
        ra = s / "review_assigned.txt"
        r["assigned"] = ra.read_text(encoding="utf-8", errors="replace").strip().replace("\n", " | ") if ra.exists() else None
        r["pre"] = (s / ".pre_ingest").is_dir()
        r["pre_files"] = sorted(p.name for p in (s / ".pre_ingest").iterdir()) if r["pre"] else []
        r["rerev"] = sorted(p.name for p in (s / ".rereview").iterdir()) if (s / ".rereview").is_dir() else []
        r["dots"] = sorted(p.name for p in s.iterdir() if p.is_dir() and p.name.startswith("."))
        r["rois"] = (s / "ROIs.jpg").exists()
        r["neuron_t"] = ts(s / "neuron.mat") if (s / "neuron.mat").exists() else None
        r["movie"] = (s / (s.name + ".mat")).exists()
        r["tif"] = (s / (s.name + ".tif")).exists()
        meta = s / "review_meta.txt"
        r["meta"] = meta.read_text(encoding="utf-8", errors="replace").strip() if meta.exists() else None
        # server copy of labels
        if EXCH is not None and r.get("src"):
            sp = Path(r["src"]) / "labels.mat"
            try:
                if sp.exists() and lab_p.exists():
                    r["srv_same"] = (md5(sp) == md5(lab_p))
                    r["srv_t"] = ts(sp)
                else:
                    r["srv_same"] = None
            except Exception as e:
                r["srv_same"] = "ERR %s" % e
        rows.append(r)

print("DATA_PARENT", DATA)
print("n sessions matched:", len(rows))
print()
hdr = "%-52s %-16s %4s %4s %4s | %4s %4s %4s %3s %3s | %-7s %-11s %-11s %-5s %-4s %-4s %s"
print(hdr % ("session", "task", "N", "auto", "exp", "L", "keep", "del", "mot", "oth", "rev", "ingested", "labels_t", "srv=", "pre", "ROIs", "dot folders"))
for r in rows:
    N = r.get("N"); a = r.get("auto")
    exp = (N - a) if N is not None else None
    L = r.get("L")
    flag = ""
    if L is not None and exp is not None and L != exp:
        flag = "  <-- LENGTH MISMATCH"
    print(hdr % (r["name"].replace("AVG5x-TSeries-", ""), r["task"], N, a, exp,
                 L if L is not None else "-", r.get("keep", "-"), r.get("del", "-"),
                 r.get("motion", "-") if r.get("motion") is not None else "-", r.get("other", "-"),
                 r.get("rev", "-"), r.get("ing", "-"), r.get("lab_t", "-"),
                 {True: "same", False: "DIFF", None: "-"}.get(r.get("srv_same"), str(r.get("srv_same"))),
                 "yes" if r["pre"] else "no", "yes" if r["rois"] else "no", ",".join(r["dots"])) + flag)

print()
print("--- sessions without labels ---")
for r in rows:
    if "L" not in r:
        print("  %-52s assigned: %s" % (r["name"].replace("AVG5x-TSeries-", ""), r["assigned"]))

print()
print("--- .pre_ingest contents for sessions ingested 11-20 ---")
for r in rows:
    if r.get("ing", "").startswith("11-20"):
        print("  %-48s %s" % (r["name"].replace("AVG5x-TSeries-", ""), ",".join(r["pre_files"])))

print()
print("--- review_meta.txt for sessions ingested 11-20 ---")
for r in rows:
    if r.get("ing", "").startswith("11-20"):
        print("  ==", r["name"].replace("AVG5x-TSeries-", ""), "| labels vars:", r.get("lab_vars"))
        print("    " + (r["meta"] or "(none)").replace("\n", "\n    "))

print()
tot = [r for r in rows if r.get("ing", "").startswith("11-20")]
print("ingested 11-20: n=%d  review candidates=%d  kept=%d  deleted=%d  motion=%d" % (
    len(tot), sum(r["L"] for r in tot), sum(r["keep"] for r in tot), sum(r["del"] for r in tot),
    sum((r["motion"] or 0) for r in tot)))
print("  total candidates N=%d, auto-rejected=%d" % (sum(r["N"] for r in tot), sum(r["auto"] for r in tot)))
