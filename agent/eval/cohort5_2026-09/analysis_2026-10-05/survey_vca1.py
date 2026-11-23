"""Read-only survey of every vCA1 session: frames, estimated thresholds,
candidate count, kept count, and Cn/pnr field statistics."""
import os
import re
import sys
import numpy as np
from scipy.io import loadmat

ROOT = r"D:\Julian_CNMFe\vCA1"
BYTES_PER_FRAME = 512 * 512

rows = []
for task in sorted(os.listdir(ROOT)):
    tdir = os.path.join(ROOT, task)
    if not os.path.isdir(tdir) or task.startswith("."):
        continue
    for d in sorted(os.listdir(tdir)):
        sd = os.path.join(tdir, d)
        if not os.path.isdir(sd) or d.startswith("."):
            continue
        m = re.search(r"TSeries-(\d{6,8})-(pnb\d+|\d+)-", d)
        date = m.group(1) if m else "?"
        animal = m.group(2) if m else "?"
        tif = os.path.join(sd, d + ".tif")
        frames_tif = None
        if os.path.exists(tif):
            frames_tif = int(round(os.path.getsize(tif) / BYTES_PER_FRAME))
        mc = mp = bd = frames_log = n_log = None
        log = os.path.join(sd, "agent_run.log")
        if os.path.exists(log):
            t = open(log, encoding="utf-8", errors="replace").read()
            m1 = re.search(r"min_corr=(\d+(?:\.\d+)?), min_pnr=(\d+(?:\.\d+)?)", t)
            if m1:
                mc, mp = float(m1.group(1)), float(m1.group(2))
            m2 = re.search(r"'bd': (\d+)", t)
            if m2:
                bd = int(m2.group(1))
            m3 = re.search(r"X (\d+) frames", t)
            if m3:
                frames_log = int(m3.group(1))
            m4 = re.search(r"Done\. (\d+) candidate neurons", t)
            if m4:
                n_log = int(m4.group(1))
        N = None
        npz = os.path.join(sd, "candidate_features.npz")
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            if "n_candidates" in z.files:
                N = int(np.asarray(z["n_candidates"]).ravel()[0])
            else:
                N = int(z["feature_matrix"].shape[0])
        kept = nrev = None
        lab = os.path.join(sd, "labels.mat")
        if os.path.exists(lab):
            L = loadmat(lab)["labels"].ravel()
            kept, nrev = int((L == 1).sum()), int(L.size)
        cn90 = fcn = p90 = elig = None
        cnf, pf = os.path.join(sd, "Cn.mat"), os.path.join(sd, "pnr.mat")
        if os.path.exists(cnf) and os.path.exists(pf):
            Cn = np.asarray(loadmat(cnf)["Cn"], dtype=float)
            pnr = np.asarray(loadmat(pf)["pnr"], dtype=float)
            if Cn.shape != pnr.shape and Cn.shape == (2 * pnr.shape[0], 2 * pnr.shape[1]):
                pnr = np.kron(pnr, np.ones((2, 2)))
            ok = np.isfinite(Cn) & np.isfinite(pnr) if Cn.shape == pnr.shape else np.zeros(1, bool)
            if Cn.shape == pnr.shape and ok.any():
                c, p = Cn[ok], pnr[ok]
                cn90 = np.percentile(c, 90)
                fcn = (c > 0.8).mean() * 100
                p90 = np.percentile(p, 90)
                if mc is not None:
                    elig = ((c > mc) & (p > mp)).mean() * 100
        rows.append(dict(task=task, animal=animal, date=date, name=d,
                         frames=frames_tif, frames_log=frames_log, mc=mc, mp=mp,
                         bd=bd, N=N, n_log=n_log, kept=kept, nrev=nrev, cn90=cn90,
                         fcn=fcn, p90=p90, elig=elig, cn_shape=Cn.shape if cn90 is not None else None))


def f(v, fmt):
    return "-" if v is None else format(v, fmt)


print("task | animal | date | frames(tif) | frames(log) | min_corr | min_pnr | bd | N cand | N log | kept/reviewed | Cn p90 | %Cn>0.8 | pnr p90 | %elig | Cn shape")
for r in rows:
    print(" | ".join([r["task"], r["animal"], r["date"], f(r["frames"], "d"), f(r["frames_log"], "d"),
                      f(r["mc"], ".2f"), f(r["mp"], ".1f"), f(r["bd"], "d"), f(r["N"], "d"), f(r["n_log"], "d"),
                      ("%s/%s" % (r["kept"], r["nrev"])) if r["kept"] is not None else "-",
                      f(r["cn90"], ".3f"), f(r["fcn"], ".1f"), f(r["p90"], ".1f"), f(r["elig"], ".2f"),
                      str(r["cn_shape"])]))

print()
print("Sessions found:", len(rows))
# frame-count check: tif size vs log
bad = [r["name"] for r in rows if r["frames"] is not None and r["frames_log"] is not None and r["frames"] != r["frames_log"]]
print("tif-size frames != log frames:", bad)
badN = [r["name"] for r in rows if r["N"] is not None and r["n_log"] is not None and r["N"] != r["n_log"]]
print("npz N != log N:", badN)
