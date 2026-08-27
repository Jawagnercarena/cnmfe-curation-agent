% m03_loadonly_check.m -- attack #3, the one link Python cannot close.
% LOAD-ONLY.  Writes nothing except stdout (captured by the caller).
% For a few sessions: (1) full(neuron.A) from neuron.mat (the final curated
% set, an MCOS Sources2D object) equals spatial_footprints.mat flattened
% column-major; (2) full(rn.neuron.A) from review_neuron.mat equals the
% .feature_expansion extraction A.  max|diff| must be 0 on both.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
base = 'D:/Julian_CNMFe';
cases = { ...
  'BLA',  '2tones/AVG5x-TSeries-093025-bla21-313um-38z-000'; ...   % retro-era agent session
  'BLA',  '2tones/AVG5x-TSeries-102325-bla8-731um-23z-000'; ...    % prospective
  'vCA1', '2tones/AVG5x-TSeries-101425-pnb88-115um-35z-000'; ...   % prospective, 0 auto-rejects
  'vCA1', '6odorDualDiffRew/AVG5x-TSeries-061926-pnb97-610um-24z-000'};  % threshold-0 pnb97
n_pass = 0; n_fail = 0;
for i = 1:size(cases, 1)
    area = cases{i, 1}; rel = cases{i, 2};
    sd = fullfile(base, area, rel);
    try
        nf = load(fullfile(sd, 'neuron.mat'));
        A_final = full(nf.neuron.A);                    % pixels x N_final, MATLAB order
        sf = load(fullfile(sd, 'spatial_footprints.mat'));
        st = sf.spatial_footprints;                     % N x d1 x d2
        N = size(st, 1); d1 = size(st, 2); d2 = size(st, 3);
        A_from_stack = zeros(d1 * d2, N);
        for k = 1:N
            img = squeeze(st(k, :, :));                 % d1 x d2
            A_from_stack(:, k) = img(:);                % column-major linearization
        end
        dA_final = max(abs(A_final(:) - A_from_stack(:)));
        rn = load(fullfile(sd, 'review_neuron.mat'));
        A_rev = full(rn.neuron.A);
        ex = load(fullfile(base, area, '.feature_expansion', [strrep(rel, '/', '__') '.mat']));
        dA_rev = max(abs(A_rev(:) - full(ex.A(:))));
        dC_rev = max(abs(full(rn.neuron.C_raw(:)) - ex.C_raw(:)));
        ok = (dA_final == 0) && (dA_rev == 0) && (dC_rev == 0) && isequal(size(A_final), size(A_from_stack));
        if ok; n_pass = n_pass + 1; else; n_fail = n_fail + 1; end
        fprintf('%s %s/%s: N_final=%d d1=%d d2=%d | max|neuron.A - stack(F)|=%.3g | max|rn.A - ext.A|=%.3g | max|rn.C_raw - ext.C_raw|=%.3g\n', ...
            ternary(ok), area, rel, N, d1, d2, dA_final, dA_rev, dC_rev);
    catch err
        n_fail = n_fail + 1;
        fprintf('FAIL %s/%s: %s\n', area, rel, err.message);
    end
end
fprintf('SUMMARY: %d pass, %d fail of %d\n', n_pass, n_fail, size(cases, 1));
function s = ternary(ok)
if ok; s = 'PASS'; else; s = 'FAIL'; end
end
