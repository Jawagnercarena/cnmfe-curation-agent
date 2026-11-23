"""Read-only.  Bruker XML acquisition settings + trial structure for the six cohort
animals: the new 3odor AVG5x tifs (D:/2p data/3odor) vs their processed
6odorDualDiffRew sessions (D:/Julian_CNMFe/vCA1/6odorDualDiffRew).

Per session: bitDepth, z motor/piezo, Pockels (laserPower[0]), green PMT, zoom,
um/px, framePeriod, number of Sequences (= trials), raw frames, frames per
Sequence (median/min/max), first->last Sequence start (min), tif frame count,
raw/tif ratio and the within-trial AVG frame rate Fs = (1/framePeriod) * tif / raw.
Writes a TSV.  Reads the server (X:) only; writes only to --out.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
NEW_DIR = Path(r"D:/2p data/3odor")
sys.path.append(str(Path(__file__).resolve().parents[2]))     # agent/, for local_config
from local_config import DATA_PARENT                           # noqa: E402
OLD_DIR = Path(DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
XML_ROOT = {"3odor": Path(r"X:/Jeremy/2p data/bla-pnb 3odor"),
            "6odor": Path(r"X:/Jeremy/2p data/6odorDualDiffRew")}
KEYS = ("bitDepth", "framePeriod", "opticalZoom")
ANIMAL_RE = re.compile(r"-(pnb\d+)-")
DATE_RE = re.compile(r"TSeries-(\d{6})-")


def tif_list():
    out = []
    for t in sorted(NEW_DIR.glob("AVG5x-TSeries-*.tif")):
        m = ANIMAL_RE.search(t.name)
        if m and m.group(1) in ANIMALS:
            out.append(("3odor", t.stem, t))
    for d in sorted(p for p in OLD_DIR.iterdir() if p.is_dir()):
        m = ANIMAL_RE.search(d.name)
        if m and m.group(1) in ANIMALS:
            t = d / (d.name + ".tif")
            out.append(("6odor", d.name, t if t.exists() else None))
    return out


def tif_frames(t):
    if t is None:
        return None
    with tifffile.TiffFile(str(t)) as f:
        return int((f.imagej_metadata or {}).get("images", 0))


def first_shard(head):
    d = {}
    for k in KEYS:
        m = re.search(rb'<PVStateValue key="%s" value="([^"]+)"' % k.encode(), head)
        d[k] = m.group(1).decode() if m else ""
    m = re.search(rb'key="laserPower">\s*<IndexedValue index="0" value="([^"]+)"', head)
    d["pockels"] = m.group(1).decode() if m else ""
    m = re.search(rb'key="pmtGain">.*?description="PMT Red" />\s*<IndexedValue index="1" value="([^"]+)"', head, re.S)
    d["pmt_green"] = m.group(1).decode() if m else ""
    m = re.search(rb'key="micronsPerPixel">\s*<IndexedValue index="XAxis" value="([^"]+)"', head)
    d["um_px"] = m.group(1).decode() if m else ""
    m = re.search(rb'subindex="0" value="([^"]+)" description="Z Focus"', head)
    d["z_motor"] = m.group(1).decode() if m else ""
    m = re.search(rb'subindex="1" value="([^"]+)" description="Bruker 400', head)
    d["z_piezo"] = m.group(1).decode() if m else ""
    return d


def hms(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def xml_stats(p):
    b = p.read_bytes()
    d = first_shard(b[:400000])
    seq_pos = np.array([m.start() for m in re.finditer(rb"<Sequence ", b)])
    fr_pos = np.array([m.start() for m in re.finditer(rb"<Frame ", b)])
    times = [hms(m.group(1).decode()) for m in re.finditer(rb'<Sequence [^>]*?time="([^"]+)"', b)]
    d["n_seq"] = len(seq_pos)
    d["n_raw"] = len(fr_pos)
    if len(seq_pos):
        per = np.diff(np.searchsorted(fr_pos, np.append(seq_pos, len(b) + 1)))
        d["fps_med"], d["fps_min"], d["fps_max"] = int(np.median(per)), int(per.min()), int(per.max())
    else:
        d["fps_med"] = d["fps_min"] = d["fps_max"] = 0
    if len(times) >= 2:
        span = times[-1] - times[0]
        if span < 0:
            span += 86400
        d["span_min"] = span / 60.0
        gaps = np.diff(times)
        gaps[gaps < 0] += 86400
        d["iti_med_s"] = float(np.median(gaps))
    else:
        d["span_min"] = d["iti_med_s"] = float("nan")
    return d


def find_xml(group, stem):
    ts = stem.replace("AVG5x-", "")
    m = DATE_RE.search(ts)
    root = XML_ROOT[group] / m.group(1)
    cands = [root / ts / (ts + ".xml")]
    if ts.endswith("-000-000"):
        t2 = ts[:-4]
        cands.append(root / t2 / (t2 + ".xml"))
    for c in cands:
        if c.exists():
            return c
    # folder renamed after acquisition (100126-pnb109 4um -> 440um, 100826-pnb106 158um -> 258um):
    # the XML inside keeps the acquisition name; accept it when it is the only TSeries XML there
    for d in (root / ts, root / ts[:-4]):
        if d.is_dir():
            x = [p for p in d.glob("TSeries-*.xml") if re.fullmatch(r"TSeries-.*-\d{3}\.xml", p.name)]
            if len(x) == 1:
                return x[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cols = ["group", "session", "animal", "date", "tif_frames", "bitDepth", "z_motor", "z_piezo", "pockels",
            "pmt_green", "opticalZoom", "um_px", "framePeriod", "n_seq", "n_raw", "fps_med", "fps_min", "fps_max",
            "span_min", "iti_med_s", "raw_per_tif", "fs_avg"]
    rows = []
    for group, stem, t in tif_list():
        animal = ANIMAL_RE.search(stem).group(1)
        date = DATE_RE.search(stem).group(1)
        nf = tif_frames(t)
        x = find_xml(group, stem)
        r = {"group": group, "session": stem.replace("AVG5x-TSeries-", ""), "animal": animal, "date": date,
             "tif_frames": nf if nf is not None else ""}
        if x is None:
            r["bitDepth"] = "NO_XML"
        else:
            r.update(xml_stats(x))
            if nf:
                r["raw_per_tif"] = r["n_raw"] / nf
                r["fs_avg"] = (1.0 / float(r["framePeriod"])) * nf / r["n_raw"]     # within-trial AVG rate
        rows.append(r)
        print("%s %s seq %s raw %s tif %s" % (group, r["session"], r.get("n_seq"), r.get("n_raw"), nf), flush=True)
    with open(a.out, "w", encoding="ascii", errors="replace") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                vals.append("%.4f" % v if isinstance(v, float) else str(v))
            fh.write("\t".join(vals) + "\n")
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    sys.exit(main())
