"""Would the installed motion-correction rule have fixed a delivered movie's bad stretch?

    python raw_sample_test.py <server TSeries folder> --delivered <AVG5x tif>
        --scores <motion_metrics npz of the delivered tif> --scratch <local dir>
        [--n-bad 8] [--n-clean 4] [--out <summary dir>]

Run in the suite2p env. Reads the server folder only (opens files for reading).

1. Map the delivered movie's bad frames to trials (Prairie XML frame counts; the
   AVG5x movie averages 5-frame groups on the session's global frame grid).
2. Pick --n-bad trials inside the bad stretch and --n-clean clean trials.
3. Copy those trials, plus the reference frames suite2p takes from the WHOLE
   session (mc_rule.reference_frames, the installed code), to --scratch.
4. Register the sample twice with suite2p:
     old = ops_user.npy as it was used for every delivered movie (maxregshift 0.1,
           suite2p's own reference)
     new = the installed rule (run_suite2p.REG_OVERRIDES + mc_rule.recentred_reference)
   Each frame is registered against the reference on its own, so a sample frame
   comes out as it would in a whole-session run with the same reference.
5. Average 5-frame groups on the global grid and score delivered / old / new the
   same way: band-passed (2-15 px) Pearson r against a template built from the
   CLEAN trials of that movie (better-correlated half), quiet = p90 of r over the
   clean trials, bad = r < 0.5 x quiet, degraded = r < 0.75 x quiet. A template
   from the clean trials keeps a sample that is mostly bad from setting its own
   quiet level.
6. For frames still bad under 'new': best rigid shift against the template with
   an unrestricted search, and the correlation there (on the true overlap). A
   frame that stays far below quiet at ANY shift is not an x-y problem.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import fft as sfft

SCRIPTS = Path(r"D:\2p data\scripts")
sys.path.insert(0, str(SCRIPTS))
import mc_rule  # noqa: E402
import run_suite2p  # noqa: E402

from suite2p import default_ops  # noqa: E402
from suite2p.registration import register  # noqa: E402

FACTOR = 5
MARGIN = 48
WORKERS = 8


# --------------------------------------------------------------------------- #
# server side (read only)
# --------------------------------------------------------------------------- #

def xml_frames(src: Path) -> np.ndarray:
    xmls = sorted(src.glob("*.xml"))
    if len(xmls) != 1:
        raise SystemExit(f"expected one .xml in {src}, found {len(xmls)}")
    text = xmls[0].read_text(encoding="utf-8-sig", errors="replace")
    starts = [m.start() for m in re.finditer(r"<Sequence ", text)]
    bounds = starts + [len(text)]
    return np.array([text.count("<Frame ", bounds[i], bounds[i + 1]) for i in range(len(starts))])


def cycle_file(src: Path, tr: int) -> Path:
    hits = sorted(src.glob(f"*_Cycle{tr:05d}_Ch2_*.ome.tif"))
    if len(hits) != 1:
        raise SystemExit(f"cycle {tr}: {len(hits)} files (one multi-page file per cycle expected)")
    return hits[0]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def _grids(ly: int, lx: int):
    fy = np.fft.fftfreq(ly)[:, None].astype(np.float32)
    fx = np.fft.rfftfreq(lx)[None, :].astype(np.float32)
    return fy, fx


def bandpass(x: np.ndarray) -> np.ndarray:
    ly, lx = x.shape[-2:]
    fy, fx = _grids(ly, lx)

    def g(s):
        return np.exp(-2.0 * np.pi ** 2 * s ** 2 * (fy ** 2 + fx ** 2)).astype(np.float32)

    f = sfft.rfft2(x.astype(np.float32), workers=WORKERS)
    return sfft.irfft2(f * (g(2.0) - g(15.0)), s=(ly, lx), workers=WORKERS).astype(np.float32)


def pearson(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = MARGIN
    ac = a[:, m:-m, m:-m]
    tc = t[m:-m, m:-m]
    ac = ac - ac.mean(axis=(1, 2), keepdims=True)
    tc = tc - tc.mean()
    num = (ac * tc[None]).sum(axis=(1, 2))
    den = np.sqrt((ac ** 2).sum(axis=(1, 2)) * (tc ** 2).sum())
    return (num / np.maximum(den, 1e-12)).astype(np.float32)


def score(avg: np.ndarray, clean: np.ndarray) -> dict:
    """r of every averaged frame against a template from the clean-trial frames."""
    bp = bandpass(avg)
    c = bp[clean]
    r0 = pearson(c, c.mean(axis=0))
    tmpl = c[r0 >= np.median(r0)].mean(axis=0)
    r = pearson(bp, tmpl)
    quiet = float(np.percentile(r[clean], 90))
    return {"r": r, "quiet": quiet, "bad": r < 0.5 * quiet, "degraded": r < 0.75 * quiet,
            "template_bp": tmpl, "bp": bp}


def best_shift(bp: np.ndarray, tmpl: np.ndarray):
    """Unrestricted rigid shift of each band-passed frame onto the template, and r on the overlap."""
    n, ly, lx = bp.shape
    tf = np.conj(sfft.rfft2(tmpl))
    cross = sfft.rfft2(bp, workers=WORKERS) * tf[None]
    cross /= np.abs(cross) + 1e-6
    cc = sfft.irfft2(cross, s=(ly, lx), workers=WORKERS)
    py, px = np.divmod(cc.reshape(n, -1).argmax(axis=1), lx)
    dy = np.where(py > ly // 2, py - ly, py)
    dx = np.where(px > lx // 2, px - lx, px)
    rr = np.full(n, np.nan, np.float32)
    for k in range(n):
        y, x = int(dy[k]), int(dx[k])
        a = bp[k, max(y, 0):ly + min(y, 0), max(x, 0):lx + min(x, 0)]
        t = tmpl[max(-y, 0):ly + min(-y, 0), max(-x, 0):lx + min(-x, 0)]
        a = a[24:-24, 24:-24]
        t = t[24:-24, 24:-24]
        if a.size < 0.3 * ly * lx:
            continue
        a = a - a.mean()
        t = t - t.mean()
        rr[k] = float((a * t).sum() / max(np.sqrt((a ** 2).sum() * (t ** 2).sum()), 1e-12))
    return dy, dx, rr


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--delivered", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--n-bad", type=int, default=8)
    ap.add_argument("--n-clean", type=int, default=4)
    ap.add_argument("--min-bad-share", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "logs" / "raw_sample_3odor")
    args = ap.parse_args()

    src = args.src
    name = src.name
    work = args.scratch / name
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- 1. delivered bad frames -> trials
    nfr = xml_frames(src)
    gstart = np.concatenate([[0], np.cumsum(nfr)])
    deliv = tifffile.memmap(args.delivered, mode="r")
    n_avg = deliv.shape[0]
    z = np.load(args.scores)
    r_del = np.asarray(z["r_raw"])
    if len(r_del) != n_avg:
        raise SystemExit(f"scores have {len(r_del)} frames, delivered tif {n_avg}")
    if n_avg != int(gstart[-1]) // FACTOR:
        raise SystemExit(f"delivered tif has {n_avg} frames, XML {int(gstart[-1])} raw frames "
                         f"-> {int(gstart[-1]) // FACTOR} groups of {FACTOR}: grid assumption fails")
    bad_del = r_del < 0.5 * np.percentile(r_del, 90)
    ntr = len(nfr)
    share = np.zeros(ntr)
    for t in range(ntr):
        g0 = int(gstart[t])
        a = -(-g0 // FACTOR)
        b = int(gstart[t + 1]) // FACTOR
        share[t] = bad_del[a:b].mean() if b > a else np.nan
    print(f"{name}: {ntr} cycles, {int(gstart[-1])} raw frames ({', '.join(f'{v} x{c}' for v, c in zip(*np.unique(nfr, return_counts=True)))}); "
          f"delivered {n_avg} frames, {100 * bad_del.mean():.2f}% bad", flush=True)
    print("  bad share per cycle (%): " + " ".join(f"{100 * s:.0f}" for s in share), flush=True)

    # ---- 2. trial choice
    cand = np.flatnonzero(share >= args.min_bad_share) + 1
    clean_c = np.flatnonzero(share == 0) + 1
    if len(cand) == 0 or len(clean_c) == 0:
        raise SystemExit(f"{len(cand)} bad cycles, {len(clean_c)} clean cycles: cannot build the sample")
    pick = lambda arr, k: sorted({int(arr[i]) for i in np.linspace(0, len(arr) - 1, min(k, len(arr))).round().astype(int)})
    bad_trials = pick(cand, args.n_bad)
    clean_trials = pick(clean_c, args.n_clean)
    trials = sorted(bad_trials + clean_trials)
    print(f"  cycles with >= {100 * args.min_bad_share:.0f}% bad: {len(cand)} ({cand.min()}-{cand.max()}); clean: {len(clean_c)}", flush=True)
    print(f"  sample: bad-stretch cycles {bad_trials} (bad % "
          f"{[round(100 * share[t - 1]) for t in bad_trials]}), clean cycles {clean_trials}", flush=True)

    # ---- 3. copy the sample and the reference frames (server read only)
    t0 = time.time()
    for tr in trials:
        out = work / f"cycle{tr:03d}.npy"
        if out.is_file():
            continue
        with tifffile.TiffFile(cycle_file(src, tr), is_ome=False) as tf:
            arr = np.stack([p.asarray() for p in tf.pages])
        if arr.shape[0] != nfr[tr - 1]:
            raise SystemExit(f"cycle {tr}: {arr.shape[0]} pages, XML says {nfr[tr - 1]}")
        np.save(out, arr)
    ops_old = default_ops()
    ops_old.update(np.load(run_suite2p.USER_OPS, allow_pickle=True).item())
    ops_new, source = run_suite2p.load_base_ops(run_suite2p.USER_OPS)
    diff = sorted(k for k in set(ops_old) | set(ops_new) if repr(ops_old.get(k)) != repr(ops_new.get(k)))
    if diff != sorted(run_suite2p.REG_OVERRIDES):
        raise SystemExit(f"old and new ops differ in {diff}, expected only {sorted(run_suite2p.REG_OVERRIDES)}")
    for o in (ops_old, ops_new):
        # ops_user.npy has reg_tif on; writing registered tifs does not change the
        # registration and needs a save path, so it is off for the sample.
        o["reg_tif"] = False
        o["reg_tif_chan2"] = False
    rf = work / "ref_frames.npy"
    if not rf.is_file():
        frames, idx, npages = mc_rule.reference_frames(src, ops_new)
        if not np.array_equal(npages, nfr):
            raise SystemExit("pages per tif file differ from the XML frame counts")
        np.save(rf, frames)
        np.save(work / "ref_idx.npy", idx)
    ref_frames = np.load(rf)
    print(f"  sample + {len(ref_frames)} reference frames on scratch ({time.time() - t0:.0f} s)", flush=True)

    raw_list = [np.load(work / f"cycle{tr:03d}.npy") for tr in trials]
    nper = [a.shape[0] for a in raw_list]
    raw = np.concatenate(raw_list)
    del raw_list
    if raw.dtype != np.uint16:
        raise SystemExit(f"raw dtype {raw.dtype}, uint16 expected")
    raw = (raw // 2).astype(np.int16)              # suite2p io/tiff.py conversion
    ly, lx = raw.shape[1:]

    # 5-frame groups on the global grid: (start in sample, AVG5x index, cycle)
    groups = []
    a = 0
    for tr, n in zip(trials, nper):
        g0 = int(gstart[tr - 1])
        skip = (-g0) % FACTOR
        for j in range((n - skip) // FACTOR):
            groups.append((a + skip + j * FACTOR, (g0 + skip) // FACTOR + j, tr))
        a += n
    gidx = np.array([g[1] for g in groups])
    gcyc = np.array([g[2] for g in groups])
    is_clean = np.isin(gcyc, clean_trials)
    is_badc = np.isin(gcyc, bad_trials)
    rcyc = np.repeat(trials, nper)

    # ---- 4. register
    mov = {"delivered": np.stack([np.asarray(deliv[g], dtype=np.float32) for g in gidx])}
    regs = {}
    for tag, ops in (("old", ops_old), ("new", ops_new)):
        t0 = time.time()
        info = {}
        with contextlib.redirect_stdout(io.StringIO()):
            if tag == "old":
                ref = register.compute_reference(ref_frames.copy(), ops=ops)
            else:
                orig = mc_rule.reference_frames
                mc_rule.reference_frames = lambda s, o: (ref_frames, np.load(work / "ref_idx.npy"), nfr)
                try:
                    ref, info = mc_rule.recentred_reference(src, dict(ops))
                finally:
                    mc_rule.reference_frames = orig
                if ref is None:
                    ref = register.compute_reference(ref_frames.copy(), ops=ops)
            out = np.zeros_like(raw)
            res = register.compute_reference_and_register_frames(raw.copy(), f_align_out=out, refImg=ref, ops=ops)
        yoff, xoff, cxy = [np.asarray(v, np.float32) for v in res[4]]
        lim = int(min(round(float(ops["maxregshift"]) * min(ly, lx)), min(ly, lx) // 2))
        at_lim = (np.abs(yoff) >= lim - 1) | (np.abs(xoff) >= lim - 1)
        avg = np.stack([out[s:s + FACTOR].mean(axis=0, dtype=np.float32) for s, _, _ in groups])
        mov[tag] = avg
        regs[tag] = {"yoff": yoff, "xoff": xoff, "corrXY": cxy, "at_limit": at_lim, "limit": lim,
                     "ref_info": info, "seconds": round(time.time() - t0)}
        tifffile.imwrite(work / f"{tag}_avg.tif", np.clip(avg, 0, 65535).astype(np.uint16), imagej=True)
        print(f"  registered '{tag}' in {time.time() - t0:.0f} s (search limit +/-{lim} px"
              + (f"; reference: {info.get('note')}, offset {info.get('offset')}" if tag == "new" else "") + ")",
              flush=True)
    del raw, out

    # ---- 5. score all three the same way
    sc = {tag: score(m, is_clean) for tag, m in mov.items()}
    pix = np.array([np.corrcoef(mov["old"][i, 30:-30, 30:-30].ravel(),
                                mov["delivered"][i, 30:-30, 30:-30].ravel())[0, 1] for i in range(len(gidx))])
    summary = {"session": name, "server_folder": str(src), "cycles": int(ntr), "raw_frames": int(gstart[-1]),
               "delivered_bad_pct_whole_movie": round(100 * float(bad_del.mean()), 2),
               "bad_cycles": bad_trials, "clean_cycles": clean_trials,
               "bad_cycle_delivered_bad_pct": [round(100 * float(share[t - 1]), 1) for t in bad_trials],
               "avg_frames_in_sample": int(len(gidx)), "raw_frames_in_sample": int(sum(nper)),
               "old_vs_delivered_pixel_r_median": round(float(np.median(pix)), 4),
               "old_vs_delivered_pixel_r_p05": round(float(np.percentile(pix, 5)), 4),
               "variants": {}}
    print(f"\n  old vs delivered, same frames: pixel r median {np.median(pix):.4f}, p05 {np.percentile(pix, 5):.4f} "
          f"(reproduction check)", flush=True)
    print(f"  {'movie':10s} {'quiet r':>7s} | bad-stretch cycles: {'bad%':>6s} {'degr%':>6s} | clean cycles: {'bad%':>6s} {'degr%':>6s} "
          f"| raw frames at limit: bad-stretch  clean | median shift (y,x)", flush=True)
    for tag in ("delivered", "old", "new"):
        s = sc[tag]
        row = {"quiet_r": round(s["quiet"], 4),
               "bad_stretch_bad_pct": round(100 * float(s["bad"][is_badc].mean()), 2),
               "bad_stretch_degraded_pct": round(100 * float(s["degraded"][is_badc].mean()), 2),
               "clean_bad_pct": round(100 * float(s["bad"][is_clean].mean()), 2),
               "clean_degraded_pct": round(100 * float(s["degraded"][is_clean].mean()), 2),
               "per_cycle_bad_pct": {int(t): round(100 * float(s["bad"][gcyc == t].mean()), 1) for t in trials}}
        extra = ""
        if tag in regs:
            g = regs[tag]
            rb = np.isin(rcyc, bad_trials)
            row.update({"search_limit_px": g["limit"],
                        "at_limit_pct_bad_stretch": round(100 * float(g["at_limit"][rb].mean()), 2),
                        "at_limit_pct_clean": round(100 * float(g["at_limit"][~rb].mean()), 2),
                        "median_yoff": float(np.median(g["yoff"])), "median_xoff": float(np.median(g["xoff"])),
                        "reference": {k: v for k, v in g["ref_info"].items() if k != "npages"},
                        "seconds": g["seconds"]})
            extra = (f" | {row['at_limit_pct_bad_stretch']:21.2f} {row['at_limit_pct_clean']:6.2f} "
                     f"| ({row['median_yoff']:+.0f},{row['median_xoff']:+.0f})")
        summary["variants"][tag] = row
        print(f"  {tag:10s} {s['quiet']:7.3f} | {row['bad_stretch_bad_pct']:25.2f} {row['bad_stretch_degraded_pct']:6.2f} "
              f"| {row['clean_bad_pct']:19.2f} {row['clean_degraded_pct']:6.2f}{extra}", flush=True)
    for tag in ("old", "new"):
        print(f"  per-cycle bad % ({tag}): " + "  ".join(f"{t}:{v:.0f}" for t, v in summary["variants"][tag]["per_cycle_bad_pct"].items()),
              flush=True)

    # ---- 6. what is left under 'new'
    s = sc["new"]
    left = np.flatnonzero(s["bad"])
    if len(left):
        dy, dx, rr = best_shift(s["bp"][left], s["template_bp"])
        rel = rr / s["quiet"]
        ok = np.isfinite(rel)
        summary["new_remaining_bad"] = {
            "n": int(len(left)),
            "best_shift_px_median": round(float(np.median(np.hypot(dy, dx))), 1),
            "best_shift_px_p90": round(float(np.percentile(np.hypot(dy, dx), 90)), 1),
            "recognisable_at_some_shift_pct": round(100 * float((ok & (rel >= 0.75)).mean()), 1),
            "not_recognisable_at_any_shift_pct": round(100 * float((~ok | (rel < 0.5)).mean()), 1)}
        m = summary["new_remaining_bad"]
        print(f"\n  still bad under 'new': {m['n']} frames; unrestricted best shift median {m['best_shift_px_median']} px "
              f"(p90 {m['best_shift_px_p90']}); back to >= 0.75 of quiet at some shift: {m['recognisable_at_some_shift_pct']}%; "
              f"below 0.5 of quiet at every shift: {m['not_recognisable_at_any_shift_pct']}%", flush=True)
    mf = {tag: float(np.median(mov[tag][is_badc & sc[tag]['bad']].mean(axis=(1, 2)))) if (is_badc & sc[tag]["bad"]).any() else None
          for tag in ("new",)}
    clean_mean = float(np.median(mov["new"][is_clean].mean(axis=(1, 2))))
    summary["new_mean_intensity_bad_vs_clean"] = [mf["new"], clean_mean]
    print(f"  mean intensity, 'new': still-bad frames median {mf['new']}, clean-cycle frames median {clean_mean:.1f}", flush=True)

    np.savez_compressed(work / "scores.npz", gidx=gidx, gcyc=gcyc,
                        **{f"r_{k}": v["r"] for k, v in sc.items()},
                        **{f"{k}_{f}": regs[k][f] for k in regs for f in ("yoff", "xoff", "corrXY")})
    summary["minutes"] = round((time.time() - t_start) / 60, 1)
    (args.out / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="ascii")
    print(f"\n  summary -> {args.out / (name + '.json')}  ({summary['minutes']} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
