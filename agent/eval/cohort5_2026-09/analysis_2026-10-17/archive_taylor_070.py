"""Copy Taylor's review OUTPUTS of the 0.70 bundle of 082726-pnb105 from the server
inbox (READ ONLY) into a new local dot folder, so the 0.70 review survives the 0.60
re-run (ingest will refuse it afterwards by the labels-length guard).  Copy-only."""
import sys, shutil
from pathlib import Path
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
S = "AVG5x-TSeries-082726-pnb105-391um-35z-000"
src = Path(local_config.EXCHANGE_ROOT) / "inbox" / "Taylor" / "vCA1" / "6odorDualDiffRew" / S
dst = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew" / S / ".review_mincorr070_taylor_2026-10-16"
SKIP = {S + ".mat", "review_neuron.mat", "review_report.pdf", "review_summary.txt", "Cn.mat", "pnr.mat", "Ybg_weights.mat", "run_final_review.m", "Cnn.txt"}
if dst.exists(): sys.exit("archive folder already exists: %s" % dst)
dst.mkdir()
n = 0; tot = 0
for f in sorted(src.iterdir()):
    if f.name in SKIP: continue
    if f.is_dir():
        shutil.copytree(str(f), str(dst / f.name)); n += len(list((dst / f.name).iterdir()))
    else:
        shutil.copy2(str(f), str(dst / f.name)); n += 1; tot += f.stat().st_size
(dst / "README.txt").write_text(
    "Taylor's review of the min_corr-0.70 bundle (headless run 2026-10-06 05:00, 79 candidates), returned via\n"
    "inbox/Taylor 2026-10-16 12:40 (labels.mat 79 entries, 24 kept). Copied here 2026-10-18 before the session\n"
    "was re-run at min_corr 0.60 (session override in agent/config/param_overrides.json). The matching candidate\n"
    "set is in ../.rerun_mincorr070_2026-10-06/ (spatial_footprints.mat, candidate_features.npz, Cn, pnr).\n"
    "Bundle inputs (review_neuron.mat etc.) were not copied: identical to that folder / the outbox bundle.\n", encoding="utf-8")
print("copied %d files (%.1f MB) to %s" % (n, tot / 1e6, dst))
