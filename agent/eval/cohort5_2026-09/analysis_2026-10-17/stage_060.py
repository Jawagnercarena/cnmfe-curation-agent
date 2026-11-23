"""ONE-TIME server step (user permission 2026-10-18): move the stale 0.70 outbox bundle
of 082726-pnb105 aside by RENAME (reversible; nothing deleted), after verifying it is
the 10-06 bundle (sizes == local .prev_mincorr070_2026-10-18 copies).  Then the push
is done separately with agent/push_review_bundle.py (copy-only).  --go to act."""
import sys, os
from pathlib import Path
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
S = "AVG5x-TSeries-082726-pnb105-391um-35z-000"
out = Path(local_config.EXCHANGE_ROOT) / "outbox" / "Taylor" / "vCA1" / "6odorDualDiffRew"
stale = out / S; aside = out / (".stale_mincorr070_2026-10-06_" + S)
inbox_ret = Path(local_config.EXCHANGE_ROOT) / "inbox" / "Taylor" / "vCA1" / "6odorDualDiffRew" / S
loc = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew" / S / ".prev_mincorr070_2026-10-18"
go = "--go" in sys.argv
if not stale.exists(): sys.exit("no stale bundle at %s" % stale)
if aside.exists(): sys.exit("aside folder already exists: %s" % aside)
ok = True
for f in sorted(stale.iterdir()):
    l = loc / f.name
    if f.name == S + ".mat" or f.name == "run_final_review.m": print("  %-28s %12d  (not compared)" % (f.name, f.stat().st_size)); continue
    if not l.exists():
        l = inbox_ret / f.name; src = "Taylor inbox return copy"
    else:
        src = "local 10-06 copy"
    same = l.exists() and l.stat().st_size == f.stat().st_size
    ok &= same; print("  %-28s %12d  %s: %s" % (f.name, f.stat().st_size, src, "same size" if same else "DIFFERENT/MISSING"))
if not ok: sys.exit("stale bundle does not match the 10-06 copies -- not touching it")
print("stale bundle verified as the 10-06 (0.70) bundle.")
if go:
    os.rename(str(stale), str(aside)); print("RENAMED on server:\n  %s\n  -> %s" % (stale, aside))
else:
    print("dry run: would rename\n  %s\n  -> %s" % (stale, aside))
