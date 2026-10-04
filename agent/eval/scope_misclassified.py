"""
Scope the misclassification: pre-agent (bootstrap-provenance) sessions that lack
bootstrap_match_stats.json and are therefore trained as AGENT (4.0x weight, no
ambiguous masking, no bad-session down-weighting).

Discriminators (from file-inventory comparison):
  trained_as_bootstrap = bootstrap_match_stats.json exists
  agent_provenance     = has ROIs_candidates.jpg / agent_run.log / review_report.pdf
  bootstrap_provenance = has neuron.mat (curated prior) and NO agent artifacts
MISCLASSIFIED = has labels.mat, no stats-json, bootstrap provenance.
"""
import sys
import numpy as np
import scipy.io as sio
from pathlib import Path

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
from config import DATA_ROOT

AGENT_ARTIFACTS = ["ROIs_candidates.jpg", "agent_run.log", "review_report.pdf",
                   "review_assigned.txt", "review_summary.txt"]

def n_real(sd):
    try:
        m = sio.loadmat(sd / "labels.mat")
        for k in ("labels", "label", "y"):
            if k in m:
                v = np.asarray(m[k]).ravel()
                return int((v == 1).sum()), int(v.size)
    except Exception:
        pass
    return -1, -1

rows = []
for td in sorted(DATA_ROOT.iterdir()):
    if not td.is_dir() or td.name.startswith("."):
        continue
    for sd in sorted(td.iterdir()):
        if not sd.is_dir() or not (sd / "labels.mat").exists():
            continue
        has_json  = (sd / "bootstrap_match_stats.json").exists()
        has_feat  = (sd / "candidate_features.npz").exists()
        agent_art = any((sd / a).exists() for a in AGENT_ARTIFACTS)
        has_neuron = (sd / "neuron.mat").exists()
        trained = "bootstrap" if has_json else "agent"
        prov = "agent" if agent_art else ("bootstrap" if has_neuron else "?")
        mis = (not has_json) and (not agent_art) and has_neuron and has_feat
        rows.append(dict(task=td.name, name=sd.name, has_json=has_json, has_feat=has_feat,
                         agent_art=agent_art, has_neuron=has_neuron,
                         trained=trained, prov=prov, mis=mis))

# summary
tot = len(rows)
tr_bs = sum(r["has_json"] for r in rows)
tr_ag = tot - tr_bs
mis = [r for r in rows if r["mis"]]
print(f"labeled sessions: {tot}   trained-as-bootstrap={tr_bs}   trained-as-agent={tr_ag}")
print(f"MISCLASSIFIED (bootstrap provenance, trained as agent): {len(mis)}\n")

print(f"{'task':<16}{'session':<44}{'feat':>5}{'json':>5}{'agentArt':>9}{'real/tot':>12}")
print("-" * 92)
for r in sorted(mis, key=lambda z: z["name"]):
    nr, nt = n_real(DATA_ROOT / r["task"] / r["name"])
    print(f"{r['task'][:15]:<16}{r['name'][:43]:<44}{'Y' if r['has_feat'] else 'n':>5}"
          f"{'Y' if r['has_json'] else 'n':>5}{'Y' if r['agent_art'] else 'n':>9}"
          f"{nr:>6}/{nt:<5}")

# also: any agent-provenance sessions WITHOUT json (true agent, correctly classified) for contrast count
true_agent = [r for r in rows if not r["has_json"] and r["agent_art"]]
print(f"\ncontrast: true agent-reviewed sessions (no json, has agent artifacts) = {len(true_agent)}")
orphan = [r for r in rows if not r["has_json"] and not r["agent_art"] and not r["has_neuron"]]
print(f"orphans (no json, no agent artifacts, no neuron.mat) = {len(orphan)}: "
      f"{[r['name'] for r in orphan]}")
