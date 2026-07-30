% extract_cand_traces.m -- pull the candidate trace matrix (C_raw, N x T) out of
% the Sources2D object in review_neuron.mat for each motion session, as a plain
% array, so Python can compute the population-coherence features. No video load,
% so this is fast.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
base = 'D:/Julian_CNMFe/BLA';
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
    sd = fullfile(base, sessions{i});
    try
        rn = load(fullfile(sd, 'review_neuron.mat'));
        C_raw = full(rn.neuron.C_raw);   % N x T candidate traces
        save(fullfile(sd, 'cand_traces.mat'), 'C_raw', '-v7');
        fprintf('%s: C_raw %dx%d saved\n', sessions{i}, size(C_raw, 1), size(C_raw, 2));
    catch err
        fprintf(2, 'FAILED %s: %s\n', sessions{i}, err.message);
    end
end
fprintf('done\n');
