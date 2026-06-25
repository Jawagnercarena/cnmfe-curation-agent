"""
train_merger.py
Trains a pairwise "should merge" classifier from logged manual merge decisions.

STATUS: STUB — not yet implemented.
Run this after accumulating merge_log_*.mat files from several reviewed sessions.

HOW MERGE LOGS ARE CREATED
---------------------------
CNMFe_final_save.m saves a merge_log_HHMMSS.mat to the session folder after
each manual merge step.  Each file contains:
  - neuron_bk : Sources2D object — full neuron set BEFORE the merge
  - ind_del   : (N,1) logical — which neurons were merged away (deleted)

WHAT THIS SCRIPT WILL DO (when implemented)
--------------------------------------------
1. COLLECT  — scan all session folders under D:/Julian_CNMFe/BLA/ for merge_log_*.mat
2. PAIR     — for each deleted neuron (ind_del==True), find its merge partner:
              the surviving neuron in neuron_bk with maximum spatial overlap.
              These (deleted, partner) pairs are positive "should merge" examples.
3. FEATURES — compute pairwise features for each positive pair and an equal
              number of sampled negative pairs (nearby but not merged):
                - spatial_overlap    : intersection / min(area_i, area_j)
                - trace_correlation  : Pearson r of C_raw_i and C_raw_j
                - center_distance_px : Euclidean distance between footprint CoMs
                - area_ratio         : min(area) / max(area)  (1 = same size)
                - circularity_mean   : mean circularity of the pair
                - combined_area      : area of union of the two footprints
4. TRAIN    — fit a logistic regression (or random forest) binary classifier:
              label 1 = should merge, label 0 = should not merge
5. SAVE     — write model/merger.joblib (scaler + clf)

HOW THE AGENT WILL USE IT (when implemented)
----------------------------------------------
In curator.py, after the headless run produces all candidate neurons:
  - Compute the pairwise feature matrix for all neuron pairs
  - Load merger.joblib and predict merge probability for each pair
  - Auto-merge pairs above a confidence threshold (e.g. 0.80) before
    writing review_neuron.mat — reducing the split cells the user sees

USAGE (when implemented)
--------------------------
  python agent/train_merger.py

  Options (to be added):
    --data-root   D:/Julian_CNMFe/BLA   parent data folder
    --model-out   agent/model/merger.joblib
    --threshold   0.80                  merge confidence threshold for curator
"""

# TODO: implement collect_merge_logs()
# TODO: implement extract_pairwise_features()
# TODO: implement train()
# TODO: integrate merger.joblib loading into curator.py

raise NotImplementedError(
    "train_merger.py is a stub. "
    "Implement after accumulating merge_log_*.mat files from reviewed sessions."
)
