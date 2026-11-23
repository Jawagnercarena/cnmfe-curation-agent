"""cnn_seed_variance.py -- post-hoc check (2026-09-07) of the Stage 2 CNN: training
dynamics per fold (best epoch, cap hits, loss / val-AUC curve, val-test gap) and
seed-to-seed spread of the OOF AUC vs the deployed XGBoost control, including the
8-seed-averaged OOF of each (a legitimate OOF bagging ensemble: every seed scores a
row from a model that never saw its session/animal).  Reads results/*_cnn_t8.{json,npz}
and results/*_xgb_deployed_t8.npz only; writes nothing but stdout.
Usage: python cnn_seed_variance.py > results/cnn_seed_variance.txt
"""
import json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score as auc
R = Path(r"C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent/eval/algo_bakeoff_2026-09/results")
for area in ("BLA", "vCA1"):
    for grp in ("session", "animal"):
        j = json.loads((R / f"{area}_{grp}_cnn_t8.json").read_text())
        F = j["folds"]
        be = np.array([f["best_epoch"] for f in F]); er = np.array([f["epochs_run"] for f in F])
        va = np.array([f["best_val_auc"] for f in F]); te = np.array([f["test_auc"] for f in F])
        l0 = np.array([f["curve"][0]["loss"] for f in F]); lb = np.array([f["curve"][f["best_epoch"]]["loss"] for f in F])
        ll = np.array([f["curve"][-1]["loss"] for f in F]); v0 = np.array([f["curve"][0]["val_auc"] for f in F])
        vl = np.array([f["curve"][-1]["val_auc"] for f in F])
        ntr = np.array([f["n_train"] for f in F]); nval = np.array([f["n_val"] for f in F])
        print(f"== {area}/{grp}: {len(F)} folds, epochs cap {j['epochs']}, patience {j['patience']}, lr {j['lr']}, batch {j['batch']}, runtime {j['runtime_s']/60:.0f} min")
        print(f"   n_train {ntr.min()}-{ntr.max()}  n_val(inner) {nval.min()}-{nval.max()}")
        print(f"   best_epoch: hist {np.bincount(be, minlength=25)[:25].tolist()}  median {np.median(be):.0f}  epochs_run median {np.median(er):.0f} max {er.max()}  hit cap {(er==j['epochs']).sum()}")
        print(f"   loss ep0 {l0.mean():.3f} -> at best {lb.mean():.3f} -> last {ll.mean():.3f}   val AUC ep0 {v0.mean():.4f} best {va.mean():.4f} last {vl.mean():.4f}")
        print(f"   val(best) - test AUC: mean {np.mean(va-te):+.4f}  test AUC mean {te.mean():.4f} sd {te.std():.4f} min {te.min():.4f} max {te.max():.4f}")
        # OOF: seed spread + seed-ensemble, CNN vs control on rows with raw (reviewed)
        c = np.load(R / f"{area}_{grp}_cnn_t8.npz", allow_pickle=True)
        x = np.load(R / f"{area}_{grp}_xgb_deployed_t8.npz", allow_pickle=True)
        y = c["y"].astype(int); rev = c["reviewed"].astype(bool); has = c["has_raw"]
        m = rev & has
        oc, ox = c["oof"], x["oof"]
        ac = np.array([auc(y[m], o[m]) for o in oc]); ax = np.array([auc(y[m], o[m]) for o in ox])
        ab = np.array([auc(y[m], 0.5*(o1[m]+o2[m])) for o1, o2 in zip(oc, ox)])
        ens_c = auc(y[m], oc[:, m].mean(0)); ens_x = auc(y[m], ox[:, m].mean(0))
        ens_b = auc(y[m], 0.5*(oc[:, m].mean(0)+ox[:, m].mean(0)))
        print(f"   reviewed rows with raw N={m.sum()} (real {y[m].sum()}): per-seed AUC CNN {ac.mean():.4f} sd {ac.std():.4f} [{ac.min():.4f},{ac.max():.4f}] | XGB {ax.mean():.4f} sd {ax.std():.4f} | blend {ab.mean():.4f}")
        print(f"   8-seed-averaged OOF: CNN {ens_c:.4f} ({ens_c-ac.mean():+.4f}) | XGB {ens_x:.4f} ({ens_x-ax.mean():+.4f}) | blend {ens_b:.4f} ({ens_b-ab.mean():+.4f})")
        # score correlation CNN vs XGB (complementarity)
        r = np.mean([np.corrcoef(o1[m], o2[m])[0,1] for o1,o2 in zip(oc,ox)])
        print(f"   mean per-seed corr(CNN, XGB) on those rows: {r:.3f}")
