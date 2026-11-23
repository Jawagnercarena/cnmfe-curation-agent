"""Read-only diagnostic: mean-image cell bodies vs ACORN candidates for one session.

Usage: python missed_cells.py <session_dir> <guide_dir> <cache_dir> <min_corr> <min_pnr>
Writes only into <cache_dir>.
"""
import sys, json
import numpy as np, scipy.io as sio, tifffile
from pathlib import Path
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S, G, C = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MIN_CORR, MIN_PNR = float(sys.argv[4]), float(sys.argv[5])
BD_FULL = 20          # bd=10 in the downsampled grid
R_FULL = 7            # soma disk radius, full-res px (red blobs radius ~10)

im = np.load(C / "imgs.npz"); M = im["mean"]; SAT = im["sat"] / float(im["T"])
ds = np.load(C / "ds.npy", mmap_mode="r")     # (T2, 256, 256)
Cn = sio.loadmat(S / "Cn.mat")["Cn"]          # 512x512 (seeding Cn resized)
PNR = sio.loadmat(S / "pnr.mat")["pnr"]       # 256x256 (seeding PNR)
SF = sio.loadmat(S / "spatial_footprints.mat")["spatial_footprints"].astype(np.float32)  # (N,512,512)
cs = np.load(G / "candidates_shown.npz")
review = set(cs["review_idx"].tolist()); autorej = set(cs["auto_rejected"].tolist())
N = SF.shape[0]
assert N == int(cs["n_candidates"][0]), "candidate count mismatch"

# orientation check: footprint-weighted mean brightness vs random
fp_mask = (SF >= 0.3 * SF.reshape(N, -1).max(1)[:, None, None]).any(0)
print(f"orientation check: mean img inside footprints {M[fp_mask].mean():.1f} vs outside {M[~fp_mask].mean():.1f}; "
      f"Cn inside {Cn[fp_mask].mean():.2f} vs outside {Cn[~fp_mask].mean():.2f}")

# --- detect cell bodies in the mean image (DoG, cell scale) ---
dog = ndi.gaussian_filter(M, 3) - ndi.gaussian_filter(M, 10)
valid = np.zeros_like(M, bool); valid[BD_FULL:-BD_FULL, BD_FULL:-BD_FULL] = True
v = dog[valid]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
pk = peak_local_max(dog, min_distance=8, threshold_abs=med + 3 * mad, labels=valid.astype(int))
print(f"mean-image somata detected (DoG > 3 MAD, min dist 8 px, outside bd): {len(pk)}")

yy, xx = np.mgrid[:512, :512]
fp_lvl = SF >= 0.3 * SF.reshape(N, -1).max(1)[:, None, None]  # (N,512,512) bool

# red
red = tifffile.imread(G / "red_registered_mean.tif").astype(np.float64)
if red.ndim == 3: red = red[0]
rv = red[np.isfinite(red) & (red != 0)]
rmed = np.median(rv); rmad = 1.4826 * np.median(np.abs(rv - rmed))
redz = (red - rmed) / rmad
rb = np.genfromtxt(G / "red_blobs.csv", delimiter=",", names=True)

# trace helpers (downsampled grid)
dyy, dxx = np.mgrid[:256, :256]
def disk_ds(r, c, rad):
    return (dyy - r / 2) ** 2 + (dxx - c / 2) ** 2 <= rad ** 2

rows = []
for (r, c) in pk:
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= R_FULL ** 2
    ov = fp_lvl[:, disk].mean(1)                 # fraction of disk inside each footprint
    k = int(np.argmax(ov)); best = float(ov[k])
    cat = "none" if best < 0.3 else ("review" if k in review else "autorej")
    # seeding gates on the downsampled grid (where seeding ran)
    dd = disk_ds(r, c, R_FULL / 2)
    cn_ds = Cn[::2, ::2]
    cn_max = float(cn_ds[dd].max()); pnr_max = float(PNR[dd].max())
    gate = bool(((cn_ds >= MIN_CORR) & (PNR >= MIN_PNR) & dd).any())
    # trace: disk mean, annulus neuropil
    F = ds[:, dd].mean(1).astype(np.float64)
    ann = disk_ds(r, c, 14) & ~disk_ds(r, c, 7) & ~fp_mask[::2, ::2]
    Fnp = ds[:, ann].mean(1).astype(np.float64) if ann.sum() > 20 else np.full_like(F, np.nan)
    F0 = np.median(F)
    sig = 1.4826 * np.median(np.abs(np.diff(F) - np.median(np.diff(F)))) / np.sqrt(2)
    Fs = ndi.uniform_filter1d(F, 3)
    pnr_tr = (Fs.max() - F0) / sig
    dff99 = float(np.percentile((F - F0) / F0, 99.5))
    # neuropil-corrected
    if np.isfinite(Fnp).all():
        Fc = F - 0.7 * (Fnp - np.median(Fnp)); Fc0 = np.median(Fc)
        sigc = 1.4826 * np.median(np.abs(np.diff(Fc) - np.median(np.diff(Fc)))) / np.sqrt(2)
        Fcs = ndi.uniform_filter1d(Fc, 3)
        frac3 = float(np.mean(Fcs - Fc0 > 3 * sigc)); pnr_c = (Fcs.max() - Fc0) / sigc
        r_np = float(np.corrcoef(F, Fnp)[0, 1])
    else:
        frac3 = pnr_c = r_np = np.nan
    d_red = np.min(np.hypot(rb["row"] - r, rb["col"] - c)) if rb.size else np.inf
    rows.append(dict(r=int(r), c=int(c), cat=cat, cand=k if cat != "none" else -1, cover=round(best, 2),
                     mean=float(M[disk].mean()), dog_z=float((dog[r, c] - med) / mad),
                     sat_frac=float(SAT[disk].mean()), cn_max=cn_max, pnr_max=pnr_max, gate=gate,
                     pnr_trace=float(pnr_tr), pnr_np_corr=float(pnr_c), frac_3sig=frac3,
                     dff995=dff99, r_neuropil=r_np, red_z=float(redz[disk].mean()),
                     red_soma=bool(d_red <= 10), F0=float(F0)))

def summ(sel, name):
    if not sel: print(f"  {name}: n=0"); return
    a = lambda k: np.array([x[k] for x in sel], float)
    q = lambda k: "%.2f [%.2f-%.2f]" % (np.nanmedian(a(k)), *np.nanpercentile(a(k), [25, 75]))
    print(f"  {name}: n={len(sel)}")
    print(f"     mean brightness {q('mean')}   DoG z {q('dog_z')}   frames at 255 {q('sat_frac')}")
    print(f"     seeding Cn max {q('cn_max')}   seeding PNR max {q('pnr_max')}   pass both gates {int(a('gate').sum())}/{len(sel)}")
    print(f"     trace PNR (raw) {q('pnr_trace')}   trace PNR (neuropil-corr) {q('pnr_np_corr')}   frac frames >3sd {q('frac_3sig')}")
    print(f"     dF/F 99.5 pct {q('dff995')}   corr with neuropil {q('r_neuropil')}")
    print(f"     red soma within 10 px {int(a('red_soma').sum())}/{len(sel)}   red z {q('red_z')}")

print(f"\nthresholds used by the run: min_corr {MIN_CORR}, min_pnr {MIN_PNR}")
for cat in ["review", "autorej", "none"]:
    summ([x for x in rows if x["cat"] == cat], cat)
none = [x for x in rows if x["cat"] == "none"]
print("\n'none' split by gate:")
summ([x for x in none if x["gate"]], "none, passes both gates")
summ([x for x in none if not x["gate"]], "none, fails a gate")
nf = [x for x in none if not x["gate"]]
print(f"   of the gate failures: Cn<min_corr {sum(x['cn_max']<MIN_CORR for x in nf)}, "
      f"PNR<min_pnr {sum(x['pnr_max']<MIN_PNR for x in nf)}, both {sum((x['cn_max']<MIN_CORR) and (x['pnr_max']<MIN_PNR) for x in nf)}")
print("\nred somata (from red_blobs.csv) -> nearest mean-image soma category:")
for cat in ["review", "autorej", "none"]:
    print(f"   {cat}: {sum(1 for x in rows if x['cat']==cat and x['red_soma'])}")
# red somata coverage directly
rcov = {"review": 0, "autorej": 0, "none": 0}
for i in range(rb.size):
    r, c = int(rb["row"][i]), int(rb["col"][i])
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= R_FULL ** 2
    ov = fp_lvl[:, disk].mean(1); k = int(np.argmax(ov))
    cat = "none" if ov[k] < 0.3 else ("review" if k in review else "autorej")
    rcov[cat] += 1
print(f"red somata ({rb.size}) by candidate coverage: {rcov}")

json.dump(rows, open(C / "blobs.json", "w"), indent=1)

# coverage by mean-image conspicuousness
print("\ncoverage by mean-image contrast (DoG z):")
for lo_, hi_ in [(3, 5), (5, 8), (8, 1e9)]:
    sel = [x for x in rows if lo_ <= x["dog_z"] < hi_]
    if sel:
        n = len(sel); nr = sum(x["cat"] == "review" for x in sel); na = sum(x["cat"] == "autorej" for x in sel)
        print(f"   z {lo_}-{'inf' if hi_ > 1e8 else hi_}: n={n}  review {nr} ({100*nr/n:.0f}%)  autorej {na} ({100*na/n:.0f}%)  none {n-nr-na} ({100*(n-nr-na)/n:.0f}%)")

# nearest candidate centroid for 'none' somata (greedy suppression / merge range)
cent = np.array([ndi.center_of_mass(SF[k]) for k in range(N)])
for x in rows:
    x["d_cand"] = float(np.min(np.hypot(cent[:, 0] - x["r"], cent[:, 1] - x["c"])))
for name, sel in [("none, passes gates", [x for x in none if x["gate"]]), ("none, fails gate", [x for x in none if not x["gate"]])]:
    if sel:
        d = np.array([x["d_cand"] for x in sel])
        print(f"   {name}: nearest candidate centroid median {np.median(d):.1f} px; within 22 px (one 11x11 ds local-max window) {int((d<=22).sum())}/{len(sel)}")

from scipy.stats import mannwhitneyu
a = [x["red_z"] for x in rows if x["cat"] == "review"]; b = [x["red_z"] for x in rows if x["cat"] == "none"]
if len(a) > 3 and len(b) > 3:
    print(f"red z, none vs review: Mann-Whitney p={mannwhitneyu(b, a).pvalue:.3f} (n {len(b)} vs {len(a)})")
cnt = {c_: sum(x["cat"] == c_ for x in rows) for c_ in ["review", "autorej", "none"]}
print(f"SUMMARY {S.name}: somata {len(rows)} review {cnt['review']} autorej {cnt['autorej']} none {cnt['none']} "
      f"none_fail_gate {sum(not x['gate'] for x in none)} none_fail_Cn {sum(x['cn_max']<MIN_CORR for x in none)} "
      f"none_fail_PNR {sum(x['pnr_max']<MIN_PNR for x in none)} red_somata {rb.size} red_by_cover {rcov}")

# QC figure
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
lo, hi = np.percentile(M, [1, 99.7])
for a_ in ax: a_.imshow(M, cmap="gray", vmin=lo, vmax=hi); a_.axis("off")
ax[0].set_title("GCaMP mean (computed from movie), raw")
col = {"review": "yellow", "autorej": "orange", "none": "red"}
for x in rows:
    ax[1].add_patch(plt.Circle((x["c"], x["r"]), R_FULL + 2, fill=False, color=col[x["cat"]], lw=1.3))
for i in range(rb.size):
    ax[1].add_patch(plt.Circle((rb["col"][i], rb["row"][i]), 11, fill=False, color="cyan", lw=0.8, ls="--"))
ax[1].set_title("mean-image somata: yellow=review cand, orange=auto-rejected cand, red=no candidate; cyan dashed=red soma")
plt.tight_layout(); plt.savefig(C / "missed_cells_qc.png", dpi=110)
print("QC ->", C / "missed_cells_qc.png")
