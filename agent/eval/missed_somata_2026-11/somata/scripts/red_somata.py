"""Read-only: activity metrics at red somata + example traces. Writes only into <cache_dir>.
Usage: python red_somata.py <session_dir> <guide_dir> <cache_dir> <min_corr> <min_pnr>
"""
import sys, json
import numpy as np, scipy.io as sio
from pathlib import Path
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S, G, C = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MIN_CORR, MIN_PNR = float(sys.argv[4]), float(sys.argv[5])
R_FULL = 7
im = np.load(C / "imgs.npz"); M = im["mean"]
ds = np.load(C / "ds.npy", mmap_mode="r")
Cn = sio.loadmat(S / "Cn.mat")["Cn"]; PNR = sio.loadmat(S / "pnr.mat")["pnr"]
SF = sio.loadmat(S / "spatial_footprints.mat")["spatial_footprints"].astype(np.float32)
cs = np.load(G / "candidates_shown.npz"); review = set(cs["review_idx"].tolist())
N = SF.shape[0]
fp_lvl = SF >= 0.3 * SF.reshape(N, -1).max(1)[:, None, None]; fp_mask = fp_lvl.any(0)
dog = ndi.gaussian_filter(M, 3) - ndi.gaussian_filter(M, 10)
v = dog[20:-20, 20:-20]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
yy, xx = np.mgrid[:512, :512]; dyy, dxx = np.mgrid[:256, :256]
disk_ds = lambda r, c, rad: (dyy - r / 2) ** 2 + (dxx - c / 2) ** 2 <= rad ** 2
cn_ds = Cn[::2, ::2]

def metrics(r, c):
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= R_FULL ** 2
    ov = fp_lvl[:, disk].mean(1); k = int(np.argmax(ov))
    cat = "none" if ov[k] < 0.3 else ("review" if k in review else "autorej")
    dd = disk_ds(r, c, R_FULL / 2)
    F = ds[:, dd].mean(1).astype(np.float64)
    ann = disk_ds(r, c, 14) & ~disk_ds(r, c, 7) & ~fp_mask[::2, ::2]
    Fnp = ds[:, ann].mean(1).astype(np.float64)
    Fc = F - 0.7 * (Fnp - np.median(Fnp)); Fc0 = np.median(Fc)
    sig = 1.4826 * np.median(np.abs(np.diff(Fc) - np.median(np.diff(Fc)))) / np.sqrt(2)
    Fcs = ndi.uniform_filter1d(Fc, 3)
    return dict(r=r, c=c, cat=cat, dog_z=float((dog[r, c] - med) / mad), cn_max=float(cn_ds[dd].max()),
                pnr_max=float(PNR[dd].max()),
                gate=bool(((cn_ds >= MIN_CORR) & (PNR >= MIN_PNR) & dd).any()),
                pnr_c=float((Fcs.max() - Fc0) / sig), frac3=float(np.mean(Fcs - Fc0 > 3 * sig)),
                trace=((Fc - Fc0) / np.median(F)).astype(np.float32))

rb = np.genfromtxt(G / "red_blobs.csv", delimiter=",", names=True)
reds = [metrics(int(rb["row"][i]), int(rb["col"][i])) for i in range(np.atleast_1d(rb).size)] if rb.size else []
json.dump([{k: v for k, v in x.items() if k != "trace"} for x in reds], open(C / "reds.json", "w"), indent=1)
if not reds: print("no red somata"); sys.exit(0)
q = lambda sel, k: "%.2f [%.2f-%.2f]" % (np.median([x[k] for x in sel]), *np.percentile([x[k] for x in sel], [25, 75]))
print(f"red somata {len(reds)}  (min_corr {MIN_CORR}, min_pnr {MIN_PNR})")
for cat in ["review", "autorej", "none"]:
    sel = [x for x in reds if x["cat"] == cat]
    if not sel: print(f"  {cat}: n=0"); continue
    print(f"  {cat}: n={len(sel)}  GCaMP mean-image DoG z {q(sel,'dog_z')}  visible soma (z>=3) {sum(x['dog_z']>=3 for x in sel)}/{len(sel)}")
    print(f"      Cn max {q(sel,'cn_max')}  PNR max {q(sel,'pnr_max')}  pass gates {sum(x['gate'] for x in sel)}/{len(sel)}")
    print(f"      trace PNR (np-corr) {q(sel,'pnr_c')}  frac frames >3sd {q(sel,'frac3')}")

# example traces: GCaMP mean-image somata by category (from missed_cells.py output)
blobs = json.load(open(C / "blobs.json"))
groups = [("review candidate", [b for b in blobs if b["cat"] == "review"]),
          ("no candidate, fails Cn gate", [b for b in blobs if b["cat"] == "none" and not b["gate"]]),
          ("no candidate, passes gates", [b for b in blobs if b["cat"] == "none" and b["gate"]])]
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(18, 9), sharey=True)
T2 = ds.shape[0]; t = np.arange(T2) / (3.75 / 2) / 60.0     # pipeline Fs 3.75 / tsub 2 (time axis nominal)
for ax, (name, sel) in zip(axs, groups):
    pick = [sel[i] for i in rng.choice(len(sel), size=min(8, len(sel)), replace=False)] if sel else []
    for j, b in enumerate(pick):
        tr = metrics(b["r"], b["c"])["trace"]
        ax.plot(t, tr + 0.4 * j, lw=0.4, color="k")
        ax.text(t[-1], 0.4 * j, f" z{b['dog_z']:.0f} Cn{b['cn_max']:.2f}", fontsize=7, va="bottom")
    ax.set_title(f"{name} (8 random of {len(sel)})"); ax.set_xlabel("minutes (nominal Fs)")
axs[0].set_ylabel("neuropil-corrected dF/F, offset 0.4 per trace")
plt.suptitle(f"{S.name}: GCaMP mean-image somata, same y scale"); plt.tight_layout()
plt.savefig(C / "example_traces.png", dpi=100); print("traces ->", C / "example_traces.png")
