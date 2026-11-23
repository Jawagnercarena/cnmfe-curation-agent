"""Read-only.  Is a 3odor field the same field (same cells) as a 6odor field?

Mean image of 300 frames spread over each session (saved by video_compare.py),
difference-of-Gaussians (sigma 1.5 - 8 px, cell scale), Hann window, z-scored;
the second image is resampled to the first one's um/px when the zoom differs
(XML micronsPerPixel).  Score = peak of the zero-padded cross-correlation over
shifts up to 160 px, i.e. the correlation at the best translation.
Reference: pairs inside each task that the imaging sheet labels the same FOV
vs a different FOV.  Writes pairs.tsv and best.tsv under --out.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

MAXS = 160
WIN = np.outer(np.hanning(512), np.hanning(512))
SHEETS = {"6odor": r"D:/Analysis/6odor dual DR/2p imaging parameters 4.0 (retro) - 4_6odor DR.csv",
          "3odor": r"D:/Analysis/3odor/2p imaging parameters 4.0 (retro) - 3odor.csv"}
ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")


def fov_labels():
    lab = {}
    for g, p in SHEETS.items():
        for r in csv.DictReader(open(p, encoding="utf-8", errors="replace")):
            a = (r.get("Animal ID") or "").strip().lower()
            if a not in ANIMALS or not (r.get("z-location") or "").strip():
                continue
            m = re.findall(r"FOV\s*([A-D])", r.get("Notes") or "", re.I)
            mm, dd, yy = r["Date"].split("/")
            key = (g, a, "%02d%02d%s" % (int(mm), int(dd), yy))
            lab[key] = m[0].upper() if m else ""
    return lab


def prep(img, um, um_ref):
    img = img.astype(np.float64)
    if abs(um - um_ref) > 1e-6:
        z = ndi.zoom(img, um / um_ref, order=1)
        out = np.full((512, 512), np.median(img))
        n = z.shape[0]
        if n >= 512:
            o = (n - 512) // 2
            out = z[o:o + 512, o:o + 512]
        else:
            o = (512 - n) // 2
            out[o:o + n, o:o + n] = z
        img = out
    d = ndi.gaussian_filter(img, 1.5) - ndi.gaussian_filter(img, 8)
    d[:16, :] = d[-16:, :] = d[:, :16] = d[:, -16:] = 0      # suite2p nonrigid edge band
    d = d * WIN
    d -= d.mean()
    return d / (np.sqrt((d * d).sum()) + 1e-12)


def score(a, b):
    F = np.fft.rfft2(a, s=(1024, 1024))
    G = np.fft.rfft2(b, s=(1024, 1024))
    c = np.fft.irfft2(F * np.conj(G), s=(1024, 1024))
    c = np.fft.fftshift(c)
    sub = c[512 - MAXS:512 + MAXS + 1, 512 - MAXS:512 + MAXS + 1]
    i = np.unravel_index(np.argmax(sub), sub.shape)
    return float(sub[i]), int(i[0] - MAXS), int(i[1] - MAXS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    um = {}
    grp = {}
    for r in csv.DictReader(open(a.xml), delimiter="\t"):
        grp[r["session"]] = r["group"]
        um[r["session"]] = float(r["um_px"]) if r["um_px"] else None
    lab = fov_labels()
    sess = {}
    for p in sorted(a.img.glob("*.npz")):
        s = p.stem
        if s == "100926-pnb109-427um-25z-000":      # laser failure, deleted 2026-11-11
            continue
        an = re.search(r"-(pnb\d+)-", "-" + s).group(1)
        sess.setdefault(an, []).append(s)
    rows = []
    for an, ss in sess.items():
        imgs = {s: np.load(a.img / (s + ".npz"))["spread_mean"] for s in ss}
        for i, s1 in enumerate(ss):
            for s2 in ss[i + 1:]:
                u1, u2 = um.get(s1), um.get(s2)
                A = prep(imgs[s1], u1, u1)
                B = prep(imgs[s2], u2, u1)
                sc, dy, dx = score(A, B)
                g1, g2 = grp[s1], grp[s2]
                l1 = lab.get((g1, an, s1[:6]), "")
                l2 = lab.get((g2, an, s2[:6]), "")
                rows.append(dict(animal=an, s1=s1, g1=g1, fov1=l1, s2=s2, g2=g2, fov2=l2, score=sc, dy=dy, dx=dx,
                                 zoom_diff=int(abs(u1 - u2) > 1e-6)))
    a.out.mkdir(parents=True, exist_ok=True)
    keys = ["animal", "s1", "g1", "fov1", "s2", "g2", "fov2", "score", "dy", "dx", "zoom_diff"]
    with open(a.out / "pairs.tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join("%.4f" % r[k] if k == "score" else str(r[k]) for k in keys) + "\n")
    # reference distributions inside each task
    for g in ("6odor", "3odor"):
        same = [r["score"] for r in rows if r["g1"] == g and r["g2"] == g and r["fov1"] and r["fov1"] == r["fov2"]]
        diff = [r["score"] for r in rows if r["g1"] == g and r["g2"] == g and r["fov1"] and r["fov2"]
                and r["fov1"] != r["fov2"]]
        for nm, v in (("same-label", same), ("diff-label", diff)):
            v = np.array(v)
            if len(v):
                print("%s %s pairs N %d  score p10/p50/p90 %.3f/%.3f/%.3f  min %.3f max %.3f" % (
                    g, nm, len(v), *np.percentile(v, [10, 50, 90]), v.min(), v.max()))
    cross = np.array([r["score"] for r in rows if r["g1"] != r["g2"]])
    print("cross-task pairs N %d  score p10/p50/p90 %.3f/%.3f/%.3f max %.3f" % (
        len(cross), *np.percentile(cross, [10, 50, 90]), cross.max()))
    with open(a.out / "best.tsv", "w") as fh:
        fh.write("session\tfov_sheet\tbest_6odor\tbest_6odor_fov\tscore_6odor\tshift\tbest_3odor\tscore_3odor\n")
        for an, ss in sess.items():
            for s in ss:
                if grp[s] != "3odor":
                    continue
                def best(target):
                    c = [(r["score"], r["s2"] if r["s1"] == s else r["s1"], r) for r in rows
                         if (r["s1"] == s or r["s2"] == s) and
                         (r["g2"] if r["s1"] == s else r["g1"]) == target]
                    return max(c, key=lambda t: t[0]) if c else (float("nan"), "", None)
                b6, b3 = best("6odor"), best("3odor")
                f6 = lab.get(("6odor", an, b6[1][:6]), "") if b6[1] else ""
                sh = "%d,%d" % (b6[2]["dy"], b6[2]["dx"]) if b6[2] else ""
                fh.write("%s\t%s\t%s\t%s\t%.3f\t%s\t%s\t%.3f\n" % (s, lab.get(("3odor", an, s[:6]), ""), b6[1], f6, b6[0],
                                                                 sh, b3[1], b3[0]))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
