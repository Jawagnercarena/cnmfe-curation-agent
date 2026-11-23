"""READ-ONLY.  For the 12 cohort-5 pilot sessions: inbox return(s) (any reviewer folder),
local live state (candidates, ingested labels, provenance), and the imaging sheet's
CNMFe'd?/# cells columns.  Flags length mismatches and sheet mismatches."""
import sys, csv, re, time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
EX = Path(local_config.EXCHANGE_ROOT); DP = Path(local_config.DATA_PARENT); TASK = "vCA1/6odorDualDiffRew"; LOCAL = DP / TASK
SHEET = "D:/Analysis/6odor dual DR/2p imaging parameters 4.0 (retro) - 4_6odor DR.csv"
PILOT = ["AVG5x-TSeries-082626-pnb102-479um-3z-000", "AVG5x-TSeries-082726-pnb102-497um-3z-000",
         "AVG5x-TSeries-082626-pnb104-404um-3z-000", "AVG5x-TSeries-082726-pnb104-416um-3z-000",
         "AVG5x-TSeries-082626-pnb105-397um-35z-000", "AVG5x-TSeries-082726-pnb105-391um-35z-000",
         "AVG5x-TSeries-082626-pnb106-205um-4z-000", "AVG5x-TSeries-082726-pnb106-201um-4z-000",
         "AVG5x-TSeries-082626-pnb109-398um-26z-000", "AVG5x-TSeries-082726-pnb109-405um-25z-000",
         "AVG5x-TSeries-082626-pnb99-399um-3z-000", "AVG5x-TSeries-082826-pnb99-436um-3z-000-000"]
ts = lambda p: time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))
def labels(p):
    try:
        L = np.asarray(loadmat(str(p))["labels"]).ravel(); return int(L.size), int((L == 1).sum())
    except Exception as e: return ("ERR", str(e)[:30])
def ncand(sd):
    p = sd / "candidate_features.npz"
    return int(np.asarray(np.load(str(p), allow_pickle=True)["n_candidates"]).ravel()[0]) if p.exists() else None
# imaging sheet rows -> {(animal, mmddyy): (cnmfed, ncells, notes)}
rows = list(csv.reader(open(SHEET, encoding="utf-8", errors="replace"))); hdr = rows[0]
ci = {h: i for i, h in enumerate(hdr)}; ic = [i for i, h in enumerate(hdr) if h == "# cells"][0]; icn = ci["CNMFe'd?"]; inote = ci["Notes"]
sheet = {}
for r in rows[1:]:
    if len(r) <= ic or not r[0].strip(): continue
    m = re.match(r"(\d+)/(\d+)/(\d+)", r[1].strip())
    if not m: continue
    key = (r[0].strip().lower(), "%02d%02d%02d" % (int(m.group(1)), int(m.group(2)), int(m.group(3)) % 100))
    sheet[key] = (r[icn].strip(), r[ic].strip(), r[inote].strip()[:50])
inbox = EX / "inbox"
reviewers = [p.name for p in inbox.iterdir() if p.is_dir()]
print("session | LOCAL: live cand / labels (n,kept) / provenance | INBOX returns: reviewer (n,kept) date | SHEET: CNMFe'd? / # cells | flags")
for s in PILOT:
    sd = LOCAL / s; animal = re.search(r"-(pnb\d+)-", s).group(1); date = s.split("-")[2]
    n_live = ncand(sd)
    loc_lab = labels(sd / "labels.mat") if (sd / "labels.mat").exists() else None
    prov = (sd / "labels_provenance.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()[0][:40] if (sd / "labels_provenance.txt").exists() else "-"
    rets = []
    for who in reviewers:
        for r in (inbox / who / TASK / s, inbox / who / "6odorDualDiffRew" / s, inbox / who / s):
            if (r / "labels.mat").exists():
                rets.append((who, labels(r / "labels.mat"), ts(r / "labels.mat")))
    sh = sheet.get((animal, date), ("?", "?", ""))
    flags = []
    for who, (n, k), d in rets:
        if n == "ERR": flags.append("%s labels.mat unreadable (%s)" % (who, k)); continue
        if n != n_live: flags.append("%s return %d labels vs live %s (STALE)" % (who, n, n_live))
    kept_ok = [k for who, (n, k), d in rets if n != "ERR" and n == n_live]
    if kept_ok and sh[1] not in ("", "?"):
        try:
            if int(re.findall(r"\d+", sh[1])[0]) != kept_ok[0]: flags.append("sheet first number %s vs review kept %d" % (re.findall(r"\d+", sh[1])[0], kept_ok[0]))
        except Exception: flags.append("sheet unparsable: %s" % sh[1])
    if not rets: flags.append("NO RETURN")
    if loc_lab and loc_lab[0] != n_live: flags.append("LOCAL labels %d vs live %d" % (loc_lab[0], n_live))
    print("%s | %s / %s / %s | %s | %s / %s | %s" % (s[14:], n_live, loc_lab, prov, "; ".join("%s (%s,%s) %s" % (w, n, k, d) for w, (n, k), d in rets) or "-", sh[0], sh[1], "; ".join(flags) or "ok"))
