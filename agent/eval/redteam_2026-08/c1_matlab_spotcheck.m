% C1 spot-check: does the .feature_expansion extraction match review_neuron.mat?
% Load-only; writes nothing anywhere.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
base = 'D:/Julian_CNMFe/BLA';
ext = fullfile(base, '.feature_expansion');
sessions = { ...
    '2tones/AVG5x-TSeries-093025-bla21-313um-38z-000', ...
    '6odorDualDiffRew/AVG5x-TSeries-052726-bla37-339um-35z-000', ...
    '3odor/AVG5x-TSeries-072326-bla36-670um-34z-000'};
for i = 1:numel(sessions)
    rel = sessions{i};
    rn = load(fullfile(base, rel, 'review_neuron.mat'));
    ex = load(fullfile(ext, [strrep(rel, '/', '__') '.mat']));
    C_rn = full(rn.neuron.C_raw);
    A_rn = sparse(rn.neuron.A);
    same_n = isequal(size(C_rn), size(ex.C_raw));
    dC = max(abs(C_rn(:) - ex.C_raw(:)));
    dA = max(abs(A_rn(:) - ex.A(:)));
    fprintf('%s: N=%d T=%d size_match=%d maxdiff_C=%.3g maxdiff_A=%.3g\n', ...
        rel, size(C_rn,1), size(C_rn,2), same_n, full(dC), full(dA));
end
fprintf('spot-check done\n');
