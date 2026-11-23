"""Read-only.  bitDepth, z (motor, piezo), laser, PMT, zoom, trials for the 12 pilot
sessions and all pnb105 sessions, from the Bruker XML (first PVStateShard only)."""
import xml.etree.ElementTree as ET, os, re, glob
ROOT = "X:/Jeremy/2p data/6odorDualDiffRew"
KEYS = {"bitDepth", "laserPower", "pmtGain", "opticalZoom", "positionCurrent", "framePeriod"}
def first_shard(p):
    d = {}
    for ev, el in ET.iterparse(p, events=("end",)):
        if el.tag == "PVStateShard":
            for sv in el.iter("PVStateValue"):
                k = sv.get("key")
                if k not in KEYS: continue
                if sv.get("value") is not None: d[k] = sv.get("value")
                for iv in sv.iter("IndexedValue"): d["%s[%s]" % (k, iv.get("index"))] = iv.get("value")
                for sa in sv.iter("SubindexedValues"):
                    for siv in sa.iter("SubindexedValue"): d["%s[%s][%s]" % (k, sa.get("index"), siv.get("subindex"))] = siv.get("value")
            if "framePeriod" in d: return d
    return d
want = re.compile(r"TSeries-(0826|0827|0828|0905|0906|0907|0908|0909|0910)26-(pnb99|pnb102|pnb104|pnb105|pnb106|pnb109)-")
print("session | bitDepth | z motor | z piezo | laser | PMT1 | zoom | framePeriod")
for day in sorted(os.listdir(ROOT)):
    dd = os.path.join(ROOT, day)
    if not os.path.isdir(dd): continue
    for ts in sorted(os.listdir(dd)):
        if not want.search(ts): continue
        p = os.path.join(dd, ts, ts + ".xml")
        if not os.path.exists(p): print(ts, "| no xml"); continue
        d = first_shard(p)
        print("%s | %s | %s | %s | %s | %s | %s | %s" % (ts.replace("TSeries-", ""), d.get("bitDepth"), d.get("positionCurrent[ZAxis][0]"),
              d.get("positionCurrent[ZAxis][1]"), d.get("laserPower[0]", "")[:6], d.get("pmtGain[1]"), d.get("opticalZoom"), d.get("framePeriod", "")[:7]))
