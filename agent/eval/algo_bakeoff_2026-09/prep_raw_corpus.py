"""
prep_raw_corpus.py -- Stage 2 data prep for the raw-data CNN probe.

For every labeled candidate that has raw material on disk, build
  * a 3-channel 32x32 footprint patch at a gSig-normalised scale
      ch0  the candidate's own footprint (max-normalised)
      ch1  the session's correlation image Cn (robust z-scored per session;
           zeros + cn_present=0 when the session has no usable Cn)
      ch2  every OTHER candidate's footprint in the window (max-normalised sum),
           i.e. the neighbourhood context the nb_* features approximate
  * its raw trace C_raw, robust z-scored and resampled to the AVG5x rate
    (3.75 Hz, the reference rate features.py:171 assumes), stored full length
and key each row by (session, candidate index) so it joins the 35-column
matrix, the labels and the corpus weights 1:1.

Sources (both verified by the red team, attack a02):
  bootstrap sessions  <session>/bootstrap_candidates.npz   rows = ALL candidates,
                      pixel order C (rt.load_bootstrap_candidates)
  agent sessions      <AREA>/.feature_expansion/<task>__<session>.mat
                      A columns = the REVIEW SET only, pixel order F
                      (rt.load_extraction); auto-rejected candidates have no raw
                      data and are skipped (label 0 by fiat in the corpus)

Orientation is checked two ways before anything trains: an automatic vote (for
labeled reals, mean Cn under the footprint mask must beat mean Cn under the
TRANSPOSED mask) and a contact sheet of random footprints drawn on Cn.

Writes ONLY under <DATA_PARENT>/<AREA>/.feature_expansion/_raw_corpus/ (a dot
directory no scanner or watcher looks at) and results/ here.  Never touches a
session dir.  The pool is the bake-off's snapshot (results/pool_<AREA>.json).

Usage:  python prep_raw_corpus.py --area BLA [--limit 5] [--out-root D:/Julian_CNMFe]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage, signal, sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402  (pool snapshot enforcement + rt_lib)

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
PATCH = 32
BASE_AVG = 5                      # AVG5x = the 3.75 Hz reference rate
AVG_RE = re.compile(r"^AVG(\d+)x-")
MASK_FRAC = 0.20                  # features.py's footprint mask convention


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def avg_factor(name: str) -> int:
    m = AVG_RE.match(Path(name).name)
    return int(m.group(1)) if m else 1


def gsig_of(rec: dict, animal_params: dict):
    sd = rec["session_dir"]
    js = rt.load_json(sd)
    if js and isinstance(js.get("cnmfe_params"), dict) and js["cnmfe_params"].get("gSig"):
        return float(js["cnmfe_params"]["gSig"]), "bootstrap_match_stats"
    f = sd / "agent_run.log"
    if f.exists():
        ms = re.findall(r"Final parameters: \{'gSig': ([\d.]+)", f.read_text(errors="ignore"))
        if ms:
            return float(ms[-1]), "agent_run.log"
    ap = animal_params.get(rec["animal"] or "", {})
    if ap.get("gSig"):
        return float(ap["gSig"]), "animal_params"
    return 9.0, "default"


# ---------------------------------------------------------------------------
# raw sources
# ---------------------------------------------------------------------------

def session_raw(area: str, rec: dict):
    """Footprint accessor + traces + Cn for one session, or a reason string."""
    sd, n = rec["session_dir"], int(rec["n_cand"])
    if rec["is_bootstrap"]:
        if not (sd / "bootstrap_candidates.npz").exists():
            return "no bootstrap_candidates.npz"
        bc = rt.load_bootstrap_candidates(sd)
        A, d1, d2 = bc["A_rows_C"], bc["d1"], bc["d2"]
        if A.shape[0] != n or A.shape[1] != d1 * d2:
            return f"bootstrap A {A.shape} vs n_cand {n} / d1*d2 {d1 * d2}"
        rowmax = np.asarray(A.max(axis=1).todense()).ravel()
        rowmax[rowmax <= 0] = 1.0
        max1, max2, arg1 = top2_over_candidates_csr(sparse.diags(1.0 / rowmax) @ A)

        def img(i, _A=A, _d1=d1, _d2=d2):
            return np.asarray(_A.getrow(i).todense(), dtype=float).ravel().reshape(_d1, _d2)  # order C

        def others(i, _m1=max1, _m2=max2, _a1=arg1, _d1=d1, _d2=d2):
            # per-pixel max over the OTHER candidates' max-normalised footprints
            return np.where(_a1 == i, _m2, _m1).reshape(_d1, _d2)  # order C

        return dict(kind="bootstrap", img=img, others=others, C_raw=bc["C_raw"],
                    Cn=rt.load_cn(sd / "Cn.mat"), d1=d1, d2=d2, cand_idx=np.arange(n))
    ext = rt.load_extraction(area, rec["name"])
    if ext is None:
        return "no .feature_expansion extraction"
    A, d1, d2 = ext["A"], ext["d1"], ext["d2"]
    cand_idx = np.where(rec["reviewed"])[0]
    if A.shape[1] != len(cand_idx) or A.shape[0] != d1 * d2:
        return f"extraction A {A.shape} vs review set {len(cand_idx)} / d1*d2 {d1 * d2}"
    colmax = A.max(axis=0)
    colmax[colmax <= 0] = 1.0
    An = A / colmax                                   # (P, N)
    if An.shape[1] >= 2:
        part = np.partition(An, -2, axis=1)
        max1, max2 = part[:, -1].copy(), part[:, -2].copy()
        del part
    else:
        max1, max2 = An.max(axis=1), np.zeros(An.shape[0])
    arg1 = An.argmax(axis=1)
    del An

    def img(j, _A=A, _d1=d1, _d2=d2):
        return _A[:, j].reshape((_d1, _d2), order="F")

    def others(j, _m1=max1, _m2=max2, _a1=arg1, _d1=d1, _d2=d2):
        return np.where(_a1 == j, _m2, _m1).reshape((_d1, _d2), order="F")

    Cn = ext["Cn"] if ext["Cn"] is not None else rt.load_cn(sd / "Cn.mat")
    return dict(kind="agent", img=img, others=others, C_raw=ext["C_raw"], Cn=Cn,
                d1=d1, d2=d2, cand_idx=cand_idx)


def top2_over_candidates_csr(An, chunk=16384):
    """Per-pixel (max, second max, argmax) over the rows of a CSR (N, P) matrix,
    computed in pixel chunks so a 700-candidate session never densifies at once."""
    N, P = An.shape
    Ac = An.tocsc()
    max1 = np.zeros(P)
    max2 = np.zeros(P)
    arg1 = np.full(P, -1, dtype=np.int64)
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        D = Ac[:, s:e].toarray()
        if N >= 2:
            part = np.partition(D, -2, axis=0)
            max1[s:e], max2[s:e] = part[-1], part[-2]
        else:
            max1[s:e] = D.max(axis=0)
        arg1[s:e] = D.argmax(axis=0)
    return max1, max2, arg1


def prep_cn(Cn, d1, d2):
    """Cn -> (robust-z image at (d1,d2), status)."""
    if Cn is None:
        return None, "missing"
    Cn = np.asarray(Cn, dtype=float)
    status = "same_res"
    if Cn.shape != (d1, d2):
        status = f"resampled_from_{Cn.shape[0]}x{Cn.shape[1]}"
        Cn = ndimage.zoom(Cn, (d1 / Cn.shape[0], d2 / Cn.shape[1]), order=1)
        if Cn.shape != (d1, d2):
            Cn = np.pad(Cn, ((0, max(0, d1 - Cn.shape[0])), (0, max(0, d2 - Cn.shape[1]))))[:d1, :d2]
    med = np.nanmedian(Cn)
    mad = np.nanmedian(np.abs(Cn - med)) * 1.4826 + 1e-6
    return np.nan_to_num((Cn - med) / mad), status


# ---------------------------------------------------------------------------
# patches and traces
# ---------------------------------------------------------------------------

def crop(img, cy, cx, half):
    d1, d2 = img.shape
    out = np.zeros((2 * half, 2 * half), dtype=float)
    y0, x0 = cy - half, cx - half
    ys, ye = max(y0, 0), min(y0 + 2 * half, d1)
    xs, xe = max(x0, 0), min(x0 + 2 * half, d2)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out


_GRID = {}


def resize32(img):
    h, w = img.shape
    key = (h, w)
    if key not in _GRID:
        yy = np.linspace(0, h - 1, PATCH)
        xx = np.linspace(0, w - 1, PATCH)
        _GRID[key] = np.array(np.meshgrid(yy, xx, indexing="ij"))
    return ndimage.map_coordinates(img, _GRID[key], order=1, mode="nearest")


def make_patch(fp, Cn, others_img, gsig):
    """(3, 32, 32) float16 patch + scalar geometry, or None for an empty footprint.
    others_img: per-pixel max over the OTHER candidates' max-normalised footprints
    (bounded in [0, 1] and independent of how many candidates the session has)."""
    mx = fp.max()
    if mx <= 0:
        return None
    w = fp.sum()
    ys, xs = np.nonzero(fp)
    cy = int(round((ys * fp[ys, xs]).sum() / w))
    cx = int(round((xs * fp[ys, xs]).sum() / w))
    half = int(max(8, round(3.0 * gsig)))
    own = fp / mx
    ch0 = resize32(crop(own, cy, cx, half))
    ch1 = resize32(crop(Cn, cy, cx, half)) if Cn is not None else np.zeros((PATCH, PATCH))
    ch2 = resize32(crop(others_img, cy, cx, half))
    patch = np.stack([ch0, ch1, ch2]).astype(np.float16)
    mask = fp > MASK_FRAC * mx
    return patch, dict(cy=cy, cx=cx, half=half, area_px=int(mask.sum()), max_weight=float(mx))


def orient_vote(fp, Cn):
    """True if Cn under the footprint mask beats Cn under the transposed mask."""
    if Cn is None or fp.shape[0] != fp.shape[1]:
        return None
    mask = fp > MASK_FRAC * fp.max()
    if mask.sum() < 5:
        return None
    return bool(Cn[mask].mean() > Cn[mask.T].mean())


def prep_trace(x, avg):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    z = (x - med) / mad
    if avg != BASE_AVG:
        z = signal.resample_poly(z, up=avg, down=BASE_AVG)
    return np.clip(z, -60000, 60000).astype(np.float16)


# ---------------------------------------------------------------------------
# contact sheet
# ---------------------------------------------------------------------------

def contact_sheet(area, samples, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(3.2 * cols, 3.2 * rows * 2))
    axes = np.atleast_2d(axes)
    for k, s in enumerate(samples):
        r, c = divmod(k, cols)
        ax = axes[2 * r, c]
        Cn, fp, meta = s["Cn"], s["fp"], s["meta"]
        cy, cx, half = meta["cy"], meta["cx"], max(meta["half"] * 2, 48)
        if Cn is not None:
            ax.imshow(crop(Cn, cy, cx, half), cmap="gray", vmin=-2, vmax=6)
        else:
            ax.imshow(np.zeros((2 * half, 2 * half)), cmap="gray")
        m = crop(fp > MASK_FRAC * fp.max(), cy, cx, half).astype(float)
        ax.contour(m, levels=[0.5], colors="r", linewidths=0.8)
        ax.set_title(f"{s['kind']} y={s['y']} {s['sess'][-28:]}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2 = axes[2 * r + 1, c]
        p = s["patch"].astype(float)
        ax2.imshow(np.concatenate([p[0], np.clip(p[1] / 6, 0, 1), np.clip(p[2], 0, 1)], axis=1), cmap="viridis")
        ax2.set_title("patch: own | Cn | others", fontsize=6)
        ax2.set_xticks([])
        ax2.set_yticks([])
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    fig.suptitle(f"{area}: footprint contour (red, 20% of max) on Cn; row 2 = the 3 patch channels", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="BLA", choices=["BLA", "vCA1", "DG_AL"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N sessions (smoke test)")
    ap.add_argument("--out-root", default=str(rt.DATA_PARENT))
    ap.add_argument("--sheet-n", type=int, default=20)
    a = ap.parse_args()
    area = a.area
    out_dir = Path(a.out_root) / area / ".feature_expansion" / "_raw_corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_first{a.limit}" if a.limit else ""
    out_h5 = out_dir / f"{area}_raw_corpus{tag}.h5"
    log(f"{area}: output {out_h5}")

    animal_params = json.loads((rt.AGENT / "config" / "animal_params.json").read_text())
    recs = B.load_area_pool(area)
    if a.limit:
        recs = recs[:a.limit]

    images, traces, offsets = [], [], [0]
    meta = {k: [] for k in ("sess_idx", "cand_idx", "y", "reviewed", "is_bootstrap", "w", "gsig",
                            "avg", "T_orig", "T_new", "cy", "cx", "half", "area_px", "max_weight",
                            "cn_present")}
    sessions, manifest = [], {"area": area, "created": dt.datetime.now().isoformat(timespec="seconds"),
                              "sessions": [], "skipped": [], "orientation": {}}
    votes = {"bootstrap": [], "agent": []}
    sheet_pool = {"bootstrap": [], "agent": []}
    rng = np.random.default_rng(0)
    t0 = time.time()
    n_rows_expected = 0
    for si, rec in enumerate(recs):
        raw = session_raw(area, rec)
        if isinstance(raw, str):
            manifest["skipped"].append({"name": rec["name"], "is_bootstrap": bool(rec["is_bootstrap"]),
                                        "n_cand": int(rec["n_cand"]), "reason": raw})
            log(f"  skip {rec['name']}: {raw}")
            continue
        gsig, gsrc = gsig_of(rec, animal_params)
        avg = avg_factor(rec["name"])
        Cn, cn_status = prep_cn(raw["Cn"], raw["d1"], raw["d2"])
        n_rows = len(raw["cand_idx"])
        n_rows_expected += n_rows
        sess_id = len(sessions)
        sessions.append(rec["name"])
        n_empty = 0
        for j, ci in enumerate(raw["cand_idx"]):
            fp = raw["img"](j)
            made = make_patch(fp, Cn, raw["others"](j), gsig)
            if made is None:
                n_empty += 1
                continue
            patch, g = made
            if rec["y"][ci] == 1:
                v = orient_vote(fp, Cn)
                if v is not None:
                    votes[raw["kind"]].append(v)
            tr = prep_trace(raw["C_raw"][j], avg)
            images.append(patch)
            traces.append(tr)
            offsets.append(offsets[-1] + len(tr))
            meta["sess_idx"].append(sess_id)
            meta["cand_idx"].append(int(ci))
            meta["y"].append(int(rec["y"][ci]))
            meta["reviewed"].append(bool(rec["reviewed"][ci]))
            meta["is_bootstrap"].append(bool(rec["is_bootstrap"]))
            meta["w"].append(float(rec["w"][ci]))
            meta["gsig"].append(gsig)
            meta["avg"].append(avg)
            meta["T_orig"].append(int(raw["C_raw"].shape[1]))
            meta["T_new"].append(int(len(tr)))
            meta["cy"].append(g["cy"])
            meta["cx"].append(g["cx"])
            meta["half"].append(g["half"])
            meta["area_px"].append(g["area_px"])
            meta["max_weight"].append(g["max_weight"])
            meta["cn_present"].append(Cn is not None)
            if len(sheet_pool[raw["kind"]]) < 200 and rng.random() < 0.05:
                sheet_pool[raw["kind"]].append({"kind": raw["kind"], "y": int(rec["y"][ci]), "sess": rec["name"],
                                                "Cn": Cn, "fp": fp, "meta": g, "patch": patch})
        manifest["sessions"].append({
            "name": rec["name"], "kind": raw["kind"], "n_cand": int(rec["n_cand"]),
            "n_reviewed": int(rec["reviewed"].sum()), "n_rows": n_rows - n_empty, "n_empty": n_empty,
            "gsig": gsig, "gsig_source": gsrc, "avg": avg, "T": int(raw["C_raw"].shape[1]),
            "d1": int(raw["d1"]), "d2": int(raw["d2"]), "cn": cn_status, "animal": rec["animal"]})
        if (si + 1) % 10 == 0 or si == len(recs) - 1:
            log(f"  {si + 1}/{len(recs)} sessions, {len(images)} rows, {time.time() - t0:.0f}s")

    if not images:
        log("no rows -- nothing written")
        return
    X = np.stack(images)
    tr_flat = np.concatenate(traces)
    off = np.asarray(offsets, dtype=np.int64)
    n_empty_total = sum(s["n_empty"] for s in manifest["sessions"])
    manifest["totals"] = {
        "sessions_used": len(sessions), "sessions_skipped": len(manifest["skipped"]),
        "rows": int(len(X)), "rows_expected_from_sources": int(n_rows_expected),
        "rows_dropped_empty_footprint": int(n_empty_total),
        "join_ok": bool(len(X) + n_empty_total == n_rows_expected),
        "rows_positive": int(sum(meta["y"])), "rows_bootstrap": int(sum(meta["is_bootstrap"])),
        "rows_without_cn": int(len(meta["cn_present"]) - sum(meta["cn_present"])),
        "labeled_rows_without_raw": int(sum(s["n_cand"] for s in manifest["skipped"])
                                        + sum(s["n_cand"] - s["n_reviewed"] for s in manifest["sessions"]
                                              if s["kind"] == "agent")),
        "trace_frames": int(len(tr_flat)), "h5_bytes_est": int(X.nbytes + tr_flat.nbytes),
    }
    for kind, v in votes.items():
        manifest["orientation"][kind] = {"n_reals_checked": len(v),
                                         "frac_cn_under_mask_beats_transposed": float(np.mean(v)) if v else None}
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("images", data=X, chunks=(256, 3, PATCH, PATCH))
        h.create_dataset("trace_data", data=tr_flat, chunks=(1 << 20,))
        h.create_dataset("trace_offsets", data=off)
        for k, v in meta.items():
            h.create_dataset(k, data=np.asarray(v))
        h.create_dataset("sessions", data=np.array(sessions, dtype=h5py.string_dtype()))
        h.attrs["area"] = area
        h.attrs["patch"] = PATCH
        h.attrs["base_avg"] = BASE_AVG
        h.attrs["created"] = manifest["created"]
        h.attrs["pool_snapshot"] = str(RESULTS / f"pool_{area}.json")
    (out_dir / f"{area}_raw_corpus{tag}_manifest.json").write_text(json.dumps(manifest, indent=1))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"raw_corpus_{area}{tag}_manifest.json").write_text(json.dumps(manifest, indent=1))
    # contact sheet: half bootstrap, half agent, random
    picks = []
    for kind in ("bootstrap", "agent"):
        pool = sheet_pool[kind]
        if pool:
            idx = rng.choice(len(pool), size=min(a.sheet_n // 2, len(pool)), replace=False)
            picks += [pool[i] for i in idx]
    if picks:
        png = RESULTS / f"contact_sheet_{area}{tag}.png"
        contact_sheet(area, picks, png)
        log(f"contact sheet -> {png}")
    t = manifest["totals"]
    log(f"{area}: {t['rows']} rows from {t['sessions_used']} sessions ({t['sessions_skipped']} skipped, "
        f"{t['labeled_rows_without_raw']} labeled rows without raw data); positives {t['rows_positive']}; "
        f"join_ok={t['join_ok']}; orientation votes {manifest['orientation']}; "
        f"h5 ~{t['h5_bytes_est'] / 1e9:.2f} GB; {time.time() - t0:.0f}s")
    log(f"wrote {out_h5}")


if __name__ == "__main__":
    main()
