"""Acquisition facts from the Prairie View XML of each raw TSeries (read-only on the server).

Prints, per session: sequence (trial) count, frames per sequence, absolute start time of each
sequence, the gaps between sequences, and the main imaging state values.
"""
import re
import sys
from pathlib import Path

import numpy as np

RAW = Path(r"X:\Jeremy\2p data\6odorDualDiffRew")
SESSIONS = sys.argv[1:]

KEYS = ["framePeriod", "bitDepth", "opticalZoom", "dwellTime", "linesPerFrame", "pixelsPerLine",
        "rastersPerFrame", "activeMode", "objectiveLens"]


def hms(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


for name in SESSIONS:
    day = name.split("-")[1]
    xml = RAW / day / name / (name + ".xml")
    if not xml.is_file():
        print(f"{name}: XML not found at {xml}")
        continue
    text = xml.read_text(encoding="utf-8-sig", errors="replace")
    print(f"=== {name}  ({len(text) / 1e6:.1f} MB xml)")
    m = re.search(r'<PVScan[^>]*date="([^"]+)"', text)
    print("  scan date:", m.group(1) if m else "?")
    for k in KEYS:
        m = re.search(rf'key="{k}"\s+value="([^"]+)"', text)
        print(f"  {k}: {m.group(1) if m else '?'}")
    for label, pat in (("laserPower[0]", r'key="laserPower">\s*<IndexedValue index="0" value="([^"]+)"'),
                       ("pmtGain green", r'value="([^"]+)" description="PMT Green"'),
                       ("micronsPerPixel X", r'key="micronsPerPixel">\s*<IndexedValue index="XAxis" value="([^"]+)"')):
        m = re.search(pat, text)
        print(f"  {label}: {m.group(1) if m else '?'}")
    z = re.search(r'key="positionCurrent">.*?<SubindexedValues index="ZAxis">\s*<SubindexedValue subindex="0" value="([^"]+)"',
                  text, flags=re.S)
    print("  Z (subindex 0):", z.group(1) if z else "?")

    seqs = [(mm.start(), mm.group(1), mm.group(2)) for mm in
            re.finditer(r'<Sequence type="([^"]+)" cycle="(\d+)" time="[^"]*"', text)]
    times = [hms(mm.group(1)) for mm in re.finditer(r'<Sequence type="[^"]+" cycle="\d+" time="([^"]+)"', text)]
    bounds = [s[0] for s in seqs] + [len(text)]
    nfr = [text.count("<Frame ", bounds[i], bounds[i + 1]) for i in range(len(seqs))]
    nfr = np.array(nfr)
    times = np.array(times)
    print(f"  sequences: {len(seqs)}  type {sorted(set(s[1] for s in seqs))}  total frames {nfr.sum()}"
          f"  -> AVG5x frames {nfr.sum() // 5}")
    vals, cnts = np.unique(nfr, return_counts=True)
    print("  frames per sequence:", ", ".join(f"{v} x{c}" for v, c in zip(vals, cnts)))
    if len(times) > 1:
        gaps = np.diff(times)
        print(f"  first sequence start {times[0]:.0f}s of day, last {times[-1]:.0f}s; duration "
              f"{(times[-1] - times[0]) / 60:.1f} min")
        print(f"  gap between sequence starts: median {np.median(gaps):.1f}s  min {gaps.min():.1f}s  "
              f"max {gaps.max():.1f}s")
        big = np.where(gaps > 3 * np.median(gaps))[0]
        for i in big:
            print(f"    long gap {gaps[i]:.0f}s after sequence {i + 1} "
                  f"(raw frame {nfr[:i + 1].sum()}, AVG5x frame {nfr[:i + 1].sum() / 5:.0f})")
    np.save(Path(__file__).resolve().parent / "out_xml" / (name + "_nfr.npy"), nfr) if (
        Path(__file__).resolve().parent / "out_xml").is_dir() else None
    np.save(Path(__file__).resolve().parent / "out_xml" / (name + "_t.npy"), times) if (
        Path(__file__).resolve().parent / "out_xml").is_dir() else None
sys.exit(0)
