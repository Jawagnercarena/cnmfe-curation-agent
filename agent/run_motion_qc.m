% run_motion_qc.m -- run the motion QC flag on the 8 labeled motion sessions and
% report its self-validation (does the flag capture the real (m) tags?).
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
  '2tones/AVG5x-TSeries-101525-bla16-278um-36z-000'
};
for i = 1:numel(sessions)
    try
        motion_qc(fullfile(base, sessions{i}), repo_root);
    catch err
        fprintf(2, 'FAILED %s: %s\n', sessions{i}, err.message);
    end
end
fprintf('\nmotion_qc done for all sessions.\n');
