"""
prep_raw_corpus2.py -- Round 2 corpus: prep_raw_corpus.py's rows (same pool
snapshot, same per-candidate patch and trace code, same row order) plus, per
row, the max-correlated neighbour within features.NB_DIST = 60 px of centroid
distance (nb_row: h5 row index or -1; nb_corr; nb_dist; n_near), and a
configurable patch size.

  --tag v2       32x32 patches; images and traces must equal the round-1 corpus
                 (checked after the build by check_v2_against_round1)
  --tag v2_p48   48x48 patches (--patch 48)

The neighbour is chosen on the RAW traces of the session's corpus rows
(bootstrap: every candidate; agent: the review set -- auto-rejected candidates
have no raw data and cannot be neighbours), z-normalised per trace as
features.compute_v2b_features does; the centroid is make_patch's weighted
centroid.

Writes <root>/<AREA>/.feature_expansion/_raw_corpus/<AREA>_raw_corpus_<tag>.h5
and results/r2/raw_corpus_<AREA>_<tag>_manifest.json.  Never touches a session
dir.

Usage: python prep_raw_corpus2.py --area BLA --tag v2 --patch 32
       python prep_raw_corpus2.py --area BLA --tag v2_p48 --patch 48
       python prep_raw_corpus2.py --check --area BLA      # v2 vs round-1 corpus
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402
import prep_raw_corpus as P  # noqa: E402

rt = B.rt
R2 = HERE / os.environ.get("BAKEOFF_RESULTS", "results") / "r2"
NB_DIST = 60.0


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def neighbours(C, cents, nb_dist=NB_DIST):
    """Per candidate: the max-correlated other candidate within nb_dist px.
    C (n, T) raw traces, cents (n, 2) [cy, cx] -> (nb local index or -1, corr, dist, n_near)."""
    n = len(C)
    nb = np.full(n, -1, dtype=np.int64)
    nbc = np.zeros(n)
    nbd = np.zeros(n)
    nn_ = np.zeros(n, dtype=np.int64)
    if n < 2:
        return nb, nbc, nbd, nn_
    C = np.asarray(C, dtype=np.float32)
    Cz = C - C.mean(axis=1, keepdims=True)
    sd = Cz.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Cz = Cz / sd
    corr = (Cz @ Cz.T) / C.shape[1]
    d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
    near = (d <= nb_dist) & ~np.eye(n, dtype=bool)
    for i in range(n):
        if near[i].any():
            cand = np.where(near[i], corr[i], -np.inf)
            j = int(np.argmax(cand))
            nb[i], nbc[i], nbd[i], nn_[i] = j, float(corr[i, j]), float(d[i, j]), int(near[i].sum())
    return nb, nbc, nbd, nn_


def build(area, tag, patch, out_root, limit=0):
    P.PATCH = patch
    P._GRID.clear()
    out_dir = Path(out_root) / area / ".feature_expansion" / "_raw_corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = out_dir / f"{area}_raw_corpus_{tag}{'_first%d' % limit if limit else ''}.h5"
    log(f"{area}: output {out_h5} (patch {patch})")
    animal_params = json.loads((rt.AGENT / "config" / "animal_params.json").read_text())
    recs = B.load_area_pool(area)
    if limit:
        recs = recs[:limit]
    images, traces, offsets = [], [], [0]
    meta = {k: [] for k in ("sess_idx", "cand_idx", "y", "reviewed", "is_bootstrap", "w", "gsig",
                            "avg", "T_orig", "T_new", "cy", "cx", "half", "area_px", "max_weight",
                            "cn_present", "nb_row", "nb_corr", "nb_dist", "n_near")}
    sessions = []
    manifest = {"area": area, "tag": tag, "patch": patch, "nb_dist_px": NB_DIST,
                "created": dt.datetime.now().isoformat(timespec="seconds"), "sessions": [], "skipped": []}
    t0 = time.time()
    n_rows_expected = 0
    for si, rec in enumerate(recs):
        raw = P.session_raw(area, rec)
        if isinstance(raw, str):
            manifest["skipped"].append({"name": rec["name"], "is_bootstrap": bool(rec["is_bootstrap"]),
                                        "n_cand": int(rec["n_cand"]), "reason": raw})
            log(f"  skip {rec['name']}: {raw}")
            continue
        gsig, gsrc = P.gsig_of(rec, animal_params)
        avg = P.avg_factor(rec["name"])
        Cn, cn_status = P.prep_cn(raw["Cn"], raw["d1"], raw["d2"])
        n_rows = len(raw["cand_idx"])
        n_rows_expected += n_rows
        sess_id = len(sessions)
        sessions.append(rec["name"])
        row_start = len(images)
        n_empty = 0
        kept_j, cents = [], []
        for j, ci in enumerate(raw["cand_idx"]):
            fp = raw["img"](j)
            made = P.make_patch(fp, Cn, raw["others"](j), gsig)
            if made is None:
                n_empty += 1
                continue
            patch_arr, g = made
            tr = P.prep_trace(raw["C_raw"][j], avg)
            images.append(patch_arr)
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
            kept_j.append(j)
            cents.append((g["cy"], g["cx"]))
        # neighbours among this session's corpus rows
        if kept_j:
            C = np.asarray(raw["C_raw"])[np.asarray(kept_j)]
            nb, nbc, nbd, nn_ = neighbours(C, np.asarray(cents, dtype=float))
            meta["nb_row"] += [int(row_start + k) if k >= 0 else -1 for k in nb]
            meta["nb_corr"] += [float(v) for v in nbc]
            meta["nb_dist"] += [float(v) for v in nbd]
            meta["n_near"] += [int(v) for v in nn_]
        manifest["sessions"].append({
            "name": rec["name"], "kind": raw["kind"], "n_cand": int(rec["n_cand"]),
            "n_reviewed": int(rec["reviewed"].sum()), "n_rows": n_rows - n_empty, "n_empty": n_empty,
            "gsig": gsig, "gsig_source": gsrc, "avg": avg, "T": int(raw["C_raw"].shape[1]),
            "d1": int(raw["d1"]), "d2": int(raw["d2"]), "cn": cn_status, "animal": rec["animal"],
            "n_with_neighbour": int(sum(1 for k in nb if k >= 0)) if kept_j else 0})
        if (si + 1) % 10 == 0 or si == len(recs) - 1:
            log(f"  {si + 1}/{len(recs)} sessions, {len(images)} rows, {time.time() - t0:.0f}s")
    if not images:
        log("no rows -- nothing written")
        return
    X = np.stack(images)
    tr_flat = np.concatenate(traces)
    off = np.asarray(offsets, dtype=np.int64)
    n_empty_total = sum(s["n_empty"] for s in manifest["sessions"])
    nb_row = np.asarray(meta["nb_row"])
    manifest["totals"] = {
        "sessions_used": len(sessions), "sessions_skipped": len(manifest["skipped"]),
        "rows": int(len(X)), "rows_expected_from_sources": int(n_rows_expected),
        "rows_dropped_empty_footprint": int(n_empty_total),
        "join_ok": bool(len(X) + n_empty_total == n_rows_expected),
        "rows_positive": int(sum(meta["y"])), "rows_bootstrap": int(sum(meta["is_bootstrap"])),
        "rows_with_neighbour": int((nb_row >= 0).sum()),
        "nb_corr_quartiles": [float(q) for q in np.percentile(np.asarray(meta["nb_corr"])[nb_row >= 0], [25, 50, 75])] if (nb_row >= 0).any() else [],
        "trace_frames": int(len(tr_flat)), "h5_bytes_est": int(X.nbytes + tr_flat.nbytes),
    }
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("images", data=X, chunks=(256, 3, patch, patch))
        h.create_dataset("trace_data", data=tr_flat, chunks=(1 << 20,))
        h.create_dataset("trace_offsets", data=off)
        for k, v in meta.items():
            h.create_dataset(k, data=np.asarray(v))
        h.create_dataset("sessions", data=np.array(sessions, dtype=h5py.string_dtype()))
        h.attrs["area"] = area
        h.attrs["patch"] = patch
        h.attrs["base_avg"] = P.BASE_AVG
        h.attrs["tag"] = tag
    R2.mkdir(exist_ok=True)
    (R2 / f"raw_corpus_{area}_{tag}_manifest.json").write_text(json.dumps(manifest, indent=1, default=rt._jsonable))
    log(f"{area}/{tag}: {len(X)} rows ({manifest['totals']['rows_with_neighbour']} with a neighbour; corr quartiles "
        f"{manifest['totals']['nb_corr_quartiles']}), {len(sessions)} sessions, {time.time() - t0:.0f}s -> {out_h5}")


def check_v2_against_round1(area, out_root):
    """v2 (32 px) must reproduce the round-1 corpus row for row."""
    d = Path(out_root) / area / ".feature_expansion" / "_raw_corpus"
    with h5py.File(d / f"{area}_raw_corpus.h5", "r") as a, h5py.File(d / f"{area}_raw_corpus_v2.h5", "r") as b:
        same_img = np.array_equal(a["images"][:], b["images"][:])
        same_tr = np.array_equal(a["trace_data"][:], b["trace_data"][:]) and np.array_equal(a["trace_offsets"][:], b["trace_offsets"][:])
        same_meta = all(np.array_equal(a[k][:], b[k][:]) for k in ("sess_idx", "cand_idx", "y", "reviewed", "is_bootstrap", "w", "T_new"))
        same_sess = list(a["sessions"][:]) == list(b["sessions"][:])
        nb = b["nb_row"][:]
        ok_nb = bool(np.all((nb == -1) | (b["sess_idx"][:][np.clip(nb, 0, None)] == b["sess_idx"][:])))
    print(f"{area}: images identical {same_img}; traces identical {same_tr}; meta identical {same_meta}; "
          f"sessions identical {same_sess}; every neighbour in the same session {ok_nb}")
    return same_img and same_tr and same_meta and same_sess and ok_nb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="BLA", choices=["BLA", "vCA1", "DG_AL"])
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-root", default=str(rt.DATA_PARENT))
    ap.add_argument("--check", action="store_true", help="compare v2 with the round-1 corpus instead of building")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.check:
        ok = check_v2_against_round1(a.area, a.out_root)
        sys.exit(0 if ok else 1)
    build(a.area, a.tag, a.patch, a.out_root, a.limit)


if __name__ == "__main__":
    main()
