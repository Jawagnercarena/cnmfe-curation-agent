"""
c_rebalance_remaining.py — after worker A is stopped at a session boundary,
list the BLA bootstrap sessions still carrying a legacy (schema v1) JSON and
write c_bla_remaining.txt for a two-worker relaunch:

  bootstrap_preagent.py --sessions-file c_bla_remaining.txt --keep-candidates --worker 0 --num-workers 2
  bootstrap_preagent.py --sessions-file c_bla_remaining.txt --keep-candidates --worker 1 --num-workers 2

Refuses to run if any bootstrap worker python or a large MATLAB is still alive.
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(r"D:\Julian_CNMFe\BLA")
HERE = Path(__file__).parent

ps = subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
     "Where-Object { $_.CommandLine -match 'bootstrap_preagent' }).Count; "
     "(Get-Process -Name MATLAB -ErrorAction SilentlyContinue | "
     "Where-Object { $_.WorkingSet64 -gt 2GB }).Count"],
    capture_output=True, text=True).stdout.split()
n_py, n_ml = (int(x) for x in ps[:2]) if len(ps) >= 2 else (99, 99)
if n_py or n_ml:
    sys.exit(f"REFUSING: {n_py} bootstrap python + {n_ml} big MATLAB still running")

remaining = []
for jp in sorted(ROOT.rglob("bootstrap_match_stats.json")):
    if any(p.startswith(".") for p in jp.parts):
        continue
    if json.load(open(jp)).get("schema_version", 1) < 2:
        remaining.append(jp.parent)
out = HERE / "c_bla_remaining.txt"
out.write_text("# BLA sessions still on legacy JSON after worker A stop\n"
               + "\n".join(str(p) for p in remaining) + "\n", encoding="utf-8")
print(f"{len(remaining)} remaining -> {out}")
for p in remaining:
    stray = p / "retro_final.mat"
    if stray.exists():
        print(f"  NOTE stray temp from interrupted step 1: {stray}")
    if (p / "_bootstrap").exists():
        print(f"  NOTE leftover _bootstrap/ (will be overwritten): {p.name}")
