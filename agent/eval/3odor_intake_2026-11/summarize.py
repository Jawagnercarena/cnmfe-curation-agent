"""Read-only.  Join xml_compare.tsv + video_compare.tsv (+ fov best.tsv if present)
and print the 3odor-vs-6odor comparison: whole-group, per animal, per 3odor period,
and one row per 3odor session.  Writes merged.tsv under --out."""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SHEET3 = r"D:/Analysis/3odor/2p imaging parameters 4.0 (retro) - 3odor.csv"
EXCLUDE = ("100926-pnb109-427um-25z-000",)
CATS = ("before 9/29", "dim/hazy noted", "9/29-10/1 no dim note", "after cleaning")


def sheet_notes():
    """3odor imaging sheet: (animal, MMDDYY) -> (z, note, category, lumen flag)."""
    out = {}
    for r in csv.DictReader(open(SHEET3, encoding="utf-8", errors="replace")):
        an = (r.get("Animal ID") or "").strip().lower()
        z = (r.get("z-location") or "").strip()
        if not z or not (r.get("Date") or "").strip():
            continue
        mm, dd, yy = r["Date"].split("/")
        key = "%02d%02d%s" % (int(mm), int(dd), yy)
        n = (r.get("Notes") or "").lower()
        order = (int(yy), int(mm), int(dd))
        if re.search(r"dim|hazy|damaged|degraded|suck", n) and "less dim" not in n:
            cat = "dim/hazy noted"
        elif order < (26, 9, 29):
            cat = "before 9/29"
        elif "clean" in n or order >= (26, 10, 2):
            cat = "after cleaning"
        else:
            cat = "9/29-10/1 no dim note"
        out[(an, key)] = (z, r.get("Notes") or "", cat, int("lumen" in n))
    return out
M = [("frames", "frames", "%.0f"), ("n_seq", "trials", "%.0f"), ("fps_med", "raw fr/trial", "%.0f"),
     ("iti_med_s", "trial start-to-start s", "%.1f"), ("pockels", "Pockels", "%.0f"), ("pmt_green", "PMT green", "%.0f"),
     ("full_cn_p50", "Cn p50", "%.3f"), ("full_cn_p90", "Cn p90", "%.3f"), ("full_pct070", "% px Cn>0.70", "%.1f"),
     ("full_pct080", "% px Cn>0.80", "%.1f"), ("full_npk070", "Cn maxima >0.70", "%.0f"),
     ("full_npk080", "Cn maxima >0.80", "%.0f"), ("full_pnr_p90", "PNR p90 (full)", "%.1f"),
     ("blk_cn_p90", "Cn p90 (1000-fr block)", "%.3f"), ("blk_pnr_p85", "PNR p85 (block ~ min_pnr est)", "%.1f"),
     ("noise_rel", "noise / mean-image range", "%.3f"), ("noise_u8", "noise (8-bit units)", "%.1f"),
     ("pct0", "% px at 0", "%.3f"), ("pct255", "% px at 255", "%.3f"), ("mot_blk_max", "motion max, block (px)", "%.2f"),
     ("mot_spr_p95", "motion p95, spread (px)", "%.2f"), ("mot_spr_max", "motion max, spread (px)", "%.2f"),
     ("score_6odor", "best match to a 6odor field", "%.3f"), ("n_cand", "candidates (6odor)", "%.0f")]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def fmt(vals, f):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        return "-"
    return (f + " [" + f + "-" + f + "] n%d") % (np.median(v), v.min(), v.max(), len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--fov", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    xr = {r["session"]: r for r in csv.DictReader(open(a.xml), delimiter="\t")}
    vr = {r["session"]: r for r in csv.DictReader(open(a.video), delimiter="\t")}
    # 2026-11-11: 100926-pnb109 laser failure (user) -> tif deleted, not considered.
    # 100826-pnb106 was renamed 158um -> 258um (XML Z Focus -257.7), locally and on the server.
    for s in EXCLUDE:
        vr.pop(s, None)
        xr.pop(s, None)
    print("excluded: %s" % ", ".join(EXCLUDE))
    fr = {}
    if a.fov and a.fov.exists():
        fr = {r["session"]: r for r in csv.DictReader(open(a.fov), delimiter="\t")}
    notes = sheet_notes()
    rows = []
    for s, v in vr.items():
        r = dict(xr.get(s, {}))
        r.update(v)
        r.update(fr.get(s, {}))
        if r["group"] == "3odor":
            z, nt, cat, lum = notes.get((r["animal"], s[:6]), ("", "", "NO SHEET ROW", 0))
            r.update(sheet_z=z, note_cat=cat, lumen=lum)
        rows.append(r)
    have = {(r["animal"], r["session"][:6]) for r in rows if r["group"] == "3odor"}
    no_tif = sorted(k for k in notes if k[0] in ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
                    and k not in have)
    print("3odor sheet rows with a z-location but no tif row: %s" % (no_tif or "none"))
    for r in rows:
        if r["group"] == "3odor":
            fz = re.search(r"-(\d+)um-", r["session"])
            if fz and r.get("sheet_z") and fz.group(1) != r["sheet_z"]:
                print("z mismatch: %s file %s um, sheet %s um" % (r["session"], fz.group(1), r["sheet_z"]))
    missing = sorted(set(xr) - set(vr))
    print("video rows %d (3odor %d, 6odor %d); xml rows %d; xml sessions without video row: %d %s" % (
        len(rows), sum(r["group"] == "3odor" for r in rows), sum(r["group"] == "6odor" for r in rows), len(xr),
        len(missing), missing[:5]))
    bad = [r["session"] for r in rows if r.get("complete") != "1"]
    print("tifs failing the size check: %s" % (bad or "none"))
    a.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out / "merged.tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in sorted(rows, key=lambda r: (r["group"], r["session"])):
            fh.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    g6 = [r for r in rows if r["group"] == "6odor"]
    g3 = [r for r in rows if r["group"] == "3odor"]
    print("\n== whole group: median [min-max] n ==")
    print("%-32s | %-34s | %-34s" % ("metric", "6odor", "3odor"))
    for k, nm, f in M:
        print("%-32s | %-34s | %-34s" % (nm, fmt([num(r.get(k)) for r in g6], f), fmt([num(r.get(k)) for r in g3], f)))
    CK = ("frames", "pockels", "pmt_green", "full_cn_p90", "full_pct070", "full_npk070", "full_pnr_p90",
          "blk_cn_p90", "blk_pnr_p85", "noise_rel", "score_6odor")
    print("\n== 3odor by sheet-note category vs 6odor (median [min-max] n) ==")
    print("-- 6odor (all 40)")
    for k in CK:
        nm, f = [(n, ff) for kk, n, ff in M if kk == k][0]
        print("   %-30s %s" % (nm, fmt([num(r.get(k)) for r in g6], f)))
    for c in CATS + ("NO SHEET ROW",):
        sub = [r for r in g3 if r.get("note_cat") == c]
        if not sub:
            continue
        print("-- 3odor %s: %d sessions: %s" % (c, len(sub), ", ".join(sorted(r["session"][:13] for r in sub))))
        for k in CK:
            nm, f = [(n, ff) for kk, n, ff in M if kk == k][0]
            print("   %-30s %s" % (nm, fmt([num(r.get(k)) for r in sub], f)))
    # same animals, same weeks: dim-noted vs not, inside 9/29-10/1
    print("\n== inside 9/29-10/1 only: dim-noted vs no note (blk = length-matched 1000 frames) ==")
    for c in ("dim/hazy noted", "9/29-10/1 no dim note"):
        sub = [r for r in g3 if r.get("note_cat") == c and (r["session"][:6] in ("092926", "093026", "100126"))]
        print("-- %s: %d" % (c, len(sub)))
        for k in ("blk_cn_p90", "blk_pnr_p85", "full_cn_p90", "noise_rel"):
            nm, f = [(n, ff) for kk, n, ff in M if kk == k][0]
            print("   %-30s %s" % (nm, fmt([num(r.get(k)) for r in sub], f)))
    print("\n== per animal: 6odor vs 3odor (median [min-max] n) ==")
    for an in ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109"):
        print("-- %s" % an)
        for k in ("pockels", "full_cn_p90", "full_pct070", "full_npk070", "full_pnr_p90", "noise_rel"):
            nm, f = [(n, ff) for kk, n, ff in M if kk == k][0]
            print("   %-30s 6odor %-30s 3odor %s" % (nm, fmt([num(r.get(k)) for r in g6 if r["animal"] == an], f),
                                                   fmt([num(r.get(k)) for r in g3 if r["animal"] == an], f)))
    # does the image predict the candidate count? (6odor sessions seeded at 0.70)
    at070 = [r for r in g6 if r["animal"] != "pnb105" and not r["session"].startswith("090526-pnb99")
             and np.isfinite(num(r.get("n_cand")))]
    if len(at070) > 5:
        for k in ("full_npk070", "full_pct070", "full_cn_p90"):
            rho, p = spearmanr([num(r[k]) for r in at070], [num(r["n_cand"]) for r in at070])
            print("\n6odor at min_corr 0.70 (N %d): Spearman(%s, candidates) = %.2f (p %.2g)" % (len(at070), k, rho, p))
    print("\n== 3odor sessions ==")
    hdr = ("session", "note", "sheet", "frames", "trials", "Pockels", "PMT", "Cn p90", "%>0.70", "max>0.70",
           "PNR p90", "noise_rel", "mot max", "best 6odor match")
    print(" | ".join(hdr))
    for r in sorted(g3, key=lambda r: (r["session"][:6], r["animal"])):
        print(" | ".join([r["session"], r.get("note_cat", "") + ("+lumen" if r.get("lumen") else ""),
                          r.get("fov_sheet", ""), r["frames"], r.get("n_seq", ""),
                          "%.0f" % num(r.get("pockels")), r.get("pmt_green", ""), "%.3f" % num(r["full_cn_p90"]),
                          "%.1f" % num(r["full_pct070"]), r["full_npk070"], "%.1f" % num(r["full_pnr_p90"]),
                          "%.3f" % num(r["noise_rel"]), "%.2f" % max(num(r["mot_blk_max"]), num(r["mot_spr_max"])),
                          "%s %s (%s)" % (r.get("best_6odor", "")[:6], r.get("best_6odor_fov", ""),
                                          r.get("score_6odor", ""))]))


if __name__ == "__main__":
    sys.exit(main())
