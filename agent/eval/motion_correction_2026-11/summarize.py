"""Summary table + per-session figures from the motion_metrics npz files (scratch only)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
XML = HERE / "out_xml"
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
FS_RAW = 30.23
MATCH = sys.argv[1:] or ["pnb"]


def pct(x, q):
    return float(np.percentile(x, q))


rows = []
for npz in sorted(OUT.glob("*.npz")):
    if not any(m in npz.stem for m in MATCH):
        continue
    z = np.load(npz)
    name = npz.stem.replace("AVG5x-TSeries-", "")
    nt = int(z["nt"])
    d = np.hypot(z["dy"], z["dx"])
    r_raw, r_al = z["r_raw"], z["r_al"]
    # non-rigid remainder: inner 2x2 blocks (the outer ring holds wrapped/tapered edges)
    bd = np.hypot(z["bdy"], z["bdx"])
    inner = bd[:, 1:3, 1:3].reshape(nt, -1)
    nr_max = inner.max(axis=1)
    all_max = bd.reshape(nt, -1).max(axis=1)
    # frame-to-frame jitter (insensitive to slow drift)
    jit = np.hypot(np.diff(z["dy"]), np.diff(z["dx"]))
    row = dict(name=name, nt=nt,
               d50=pct(d, 50), d95=pct(d, 95), d99=pct(d, 99), dmax=float(d.max()),
               f05=float((d > 0.5).mean()), f1=float((d > 1.0).mean()), f2=float((d > 2.0).mean()),
               j95=pct(jit, 95), j99=pct(jit, 99),
               rr50=pct(r_raw, 50), rr05=pct(r_raw, 5), ra50=pct(r_al, 50), ra05=pct(r_al, 5),
               ra01=pct(r_al, 1),
               nr50=pct(nr_max, 50), nr95=pct(nr_max, 95), nr99=pct(nr_max, 99),
               nra95=pct(all_max, 95),
               fnr1=float((nr_max > 1.0).mean()))
    rows.append(row)

    # ---------------- figure
    stem = npz.stem.replace("AVG5x-", "")
    nfr_p = XML / (stem + "_nfr.npy")
    fig, ax = plt.subplots(3, 2, figsize=(15, 11), gridspec_kw=dict(width_ratios=[2.2, 1]))
    t = np.arange(nt)
    ax[0, 0].plot(t, z["dy"], lw=0.4, label="dy"); ax[0, 0].plot(t, z["dx"], lw=0.4, label="dx")
    ax[0, 0].set_ylabel("rigid residual (px)"); ax[0, 0].legend(loc="upper right")
    ax[0, 0].set_title(f"{name}: residual displacement vs session template")
    ax[1, 0].plot(t, r_raw, lw=0.4, color="0.6", label="as delivered")
    ax[1, 0].plot(t, r_al, lw=0.4, color="C3", label="after undoing the rigid residual")
    ax[1, 0].set_ylabel("frame-template r"); ax[1, 0].legend(loc="lower right")
    ax[2, 0].plot(t, nr_max, lw=0.4, color="C2")
    ax[2, 0].set_ylabel("non-rigid remainder, max of inner blocks (px)")
    ax[2, 0].set_xlabel("AVG5x frame")
    ax[0, 1].imshow(z["template"], cmap="gray", vmin=np.percentile(z["template"], 1),
                    vmax=np.percentile(z["template"], 99.7)); ax[0, 1].set_title("template"); ax[0, 1].axis("off")
    if nfr_p.is_file():
        nfr = np.load(nfr_p)
        edges = np.concatenate([[0], np.cumsum(nfr)]) / 5.0
        ntr = len(nfr)
        nb = 112
        img_r = np.full((ntr, nb), np.nan); img_d = np.full((ntr, nb), np.nan)
        for i in range(ntr):
            a = int(np.ceil(edges[i])); b = min(a + nb, nt)
            img_r[i, :b - a] = r_al[a:b]; img_d[i, :b - a] = d[a:b]
        ext = [0, nb * 5 / FS_RAW, ntr, 0]
        im = ax[1, 1].imshow(img_d, aspect="auto", cmap="magma", vmin=0, vmax=max(1.0, pct(d, 99)), extent=ext)
        ax[1, 1].set_title("|rigid residual| (px): trial x time in trial"); plt.colorbar(im, ax=ax[1, 1])
        im = ax[2, 1].imshow(img_r, aspect="auto", cmap="viridis", vmin=pct(r_al, 1), vmax=pct(r_al, 99), extent=ext)
        ax[2, 1].set_title("frame-template r (aligned): trial x time"); plt.colorbar(im, ax=ax[2, 1])
        ax[2, 1].set_xlabel("time in trial (s)"); ax[2, 1].set_ylabel("trial")
        row["trial_prof_d"] = np.nanmean(img_d, axis=0)
        row["trial_prof_r"] = np.nanmean(img_r, axis=0)
        row["trial_d"] = np.nanmean(img_d, axis=1)
        row["trial_r"] = np.nanmean(img_r, axis=1)
    else:
        ax[1, 1].axis("off"); ax[2, 1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / (name + "_motion.png"), dpi=70)
    plt.close(fig)

hdr = ("session", "frames", "d50", "d95", "d99", "dmax", ">0.5px", ">1px", ">2px", "jit95", "jit99",
       "r50", "r05", "rAl50", "rAl05", "rAl01", "nr50", "nr95", "nr99", "nr>1px")
print("d = rigid residual |shift| (px); jit = frame-to-frame change of it; r = frame-template Pearson "
      "(as delivered); rAl = after undoing the rigid residual; nr = non-rigid remainder (max of inner blocks, px)")
print(("%-28s %6s " + "%6s " * 18) % hdr)
for r in rows:
    print("%-28s %6d %6.2f %6.2f %6.2f %6.1f %6.3f %6.3f %6.3f %6.2f %6.2f %6.3f %6.3f %6.3f %6.3f %6.3f %6.2f %6.2f %6.2f %6.3f" % (
        r["name"], r["nt"], r["d50"], r["d95"], r["d99"], r["dmax"], r["f05"], r["f1"], r["f2"],
        r["j95"], r["j99"], r["rr50"], r["rr05"], r["ra50"], r["ra05"], r["ra01"],
        r["nr50"], r["nr95"], r["nr99"], r["fnr1"]))
np.save(HERE / "summary_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
