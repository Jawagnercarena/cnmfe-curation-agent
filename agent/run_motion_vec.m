% run_motion_vec.m -- batch the local-motion-vector extractor over the remaining
% 7 motion sessions (bla16 done as validation). Per-session try/catch.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
base      = 'D:/Julian_CNMFe/BLA';
sessions = {
  '6odorDualDiffRew/AVG5x-TSeries-061226-bla37-213um-37z-000'
  'Block_Valence/AVG5x-TSeries-070226-bla37-262um-37z-000'
  '6odorDualDiffRew/AVG5x-TSeries-061126-bla37-277um-35z-000'
  '6odorDualDiffRew/AVG5x-TSeries-060426-bla37-275um-35z-000'
  '6odorDualDiffRew/AVG5x-TSeries-052026-bla36-669um-29z-000'
  '2tones/AVG5x-TSeries-101525-bla12-660um-23z-000'
  '6odorDualDiffRew/AVG5x-TSeries-052826-bla37-216um-37z-000'
};
for i = 1:numel(sessions)
    sd = fullfile(base, sessions{i});
    fprintf('\n=== (%d/%d) %s ===\n', i, numel(sessions), sessions{i});
    try
        extract_motion_vectors(sd, repo_root);
    catch err
        fprintf(2, 'FAILED %s: %s\n', sessions{i}, err.message);
    end
end
fprintf('\nAll motion_vec extractions done.\n');
