"""Read-only.  Acquisition settings from the Bruker XML of both pnb105 sessions:
initial PVStateShard values, per-sequence z / laser / PMT changes, frame counts."""
import xml.etree.ElementTree as ET, os, sys
ROOT = "X:/Jeremy/2p data/6odorDualDiffRew"
S = [("082626", "TSeries-082626-pnb105-397um-35z-000"), ("082726", "TSeries-082726-pnb105-391um-35z-000")]
KEYS = ["laserPower", "pmtGain", "opticalZoom", "positionCurrent", "framePeriod", "dwellTime", "linesPerFrame",
        "pixelsPerLine", "objectiveLens", "laserWavelength", "micronsPerPixel", "scanLinePeriod", "rastersPerFrame", "bitDepth"]
def shard(el):
    d = {}
    for sv in el.iter("PVStateValue"):
        k = sv.get("key")
        if k not in KEYS: continue
        if sv.get("value") is not None: d[k] = sv.get("value")
        else:
            for iv in sv.iter("IndexedValue"):
                d["%s[%s]" % (k, iv.get("index"))] = iv.get("value")
            for sa in sv.iter("SubindexedValues"):
                for siv in sa.iter("SubindexedValue"):
                    d["%s[%s][%s]" % (k, sa.get("index"), siv.get("subindex"))] = siv.get("value")
    return d
for day, name in S:
    p = os.path.join(ROOT, day, name, name + ".xml")
    if not os.path.exists(p): print("MISSING", p); continue
    print("==", name, "(%.0f MB)" % (os.path.getsize(p) / 1e6))
    first = None; nseq = 0; nfr = 0; changes = []; zs = []; prev = None; tfirst = None; tlast = None
    for ev, el in ET.iterparse(p, events=("end",)):
        if el.tag == "PVStateShard" and first is None and el.getparent() if False else False: pass
        if el.tag == "Sequence":
            nseq += 1; frames = el.findall("Frame"); nfr += len(frames)
            if frames:
                if tfirst is None: tfirst = float(frames[0].get("relativeTime", 0)) + float(el.get("time", "0:0:0").split(":")[-1]) * 0
                sh = shard(frames[0]) if frames[0].find("PVStateShard") is not None else {}
                z = sh.get("positionCurrent[ZAxis][0]") or sh.get("positionCurrent[ZAxis]")
                zs.append((nseq, z, sh.get("laserPower[0]"), sh.get("pmtGain[0]"), sh.get("pmtGain[1]")))
            el.clear()
        elif el.tag == "PVStateShard" and first is None:
            d = shard(el)
            if "framePeriod" in d: first = d
    print("  sequences %d, frames %d" % (nseq, nfr))
    if first:
        for k in sorted(first): print("   %-32s %s" % (k, first[k]))
    zz = [z for _, z, _, _, _ in zs if z]
    if zz:
        zf = [float(z) for z in zz]
        print("  per-sequence Z: first %.2f, last %.2f, min %.2f, max %.2f; sequences with z change > 2 um vs first: %d of %d" % (
            zf[0], zf[-1], min(zf), max(zf), sum(abs(z - zf[0]) > 2 for z in zf), len(zf)))
    lp = sorted(set(l for _, _, l, _, _ in zs if l)); pg = sorted(set(g for _, _, _, g, _ in zs if g))
    print("  per-sequence laserPower[0] values: %s; pmtGain[0] values: %s" % (lp[:6], pg[:6]))
