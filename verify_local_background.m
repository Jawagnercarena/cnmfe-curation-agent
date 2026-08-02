%% verify_local_background.m
% Final gate on the local_background.m edit.
%
% Everything up to now tested a REIMPLEMENTATION of the inner loop inside a probe
% harness.  This calls the two real functions — the edited local_background and a
% byte-for-byte copy of the pre-edit version (local_background_orig) — on identical
% inputs, with exactly the arguments cnmfe_update_BG passes, and compares every
% value they return: the upsampled Yest, the full weights cell, and the rest of the
% results struct including the baseline b0.
%
% That covers the parts the harness never exercised: the ACTIVE_PX branch, the
% p_cutoff branch, the final imresize back to full resolution, and the Ybaseline
% correction applied on the way out.
%
%   session_dir = 'D:\Julian_CNMFe\BLA\3odor\AVG5x-TSeries-07082026-bla36-810um-25z-000';
%   ref_dir     = '<scratch>\refver';     % holds local_background_orig.m
%   run('C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\verify_local_background.m')

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
cvx_dirs = strsplit(genpath(fullfile(repo_root, 'cvx')), pathsep);
cvx_dirs = cvx_dirs(~cellfun(@isempty, cvx_dirs));
addpath(strjoin(cvx_dirs(~endsWith(cvx_dirs, [filesep 'narginchk_'])), pathsep));
clear cvx_dirs;
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('verify_local_background: session_dir must be set.');
end
if ~exist('ref_dir', 'var') || isempty(ref_dir)
    error('verify_local_background: ref_dir must point at local_background_orig.m');
end
addpath(ref_dir);
if ~exist('local_background_orig', 'file')
    error('local_background_orig not found on the path (%s)', ref_dir);
end

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');
warning('off', 'MATLAB:singularMatrix');

%% ---- build exactly the inputs cnmfe_update_BG hands to localBG ----
fprintf('[VERIFY] Loading session...\n');
ndata  = load(fullfile(session_dir, 'neuron.mat'));
neuron = ndata.neuron; clear ndata;

[~, session_nm] = fileparts(session_dir);
nam_mat = fullfile(session_dir, [session_nm, '.mat']);
if ~isfile(nam_mat)
    cand   = dir(fullfile(session_dir, '*.mat'));
    reject = {'neuron','Cn','Coor','pnr','Ybg','Ybg_mean', ...
              'spatial_footprints','review_neuron','Ybg_weights'};
    for i = numel(cand):-1:1
        [~, nm] = fileparts(cand(i).name);
        if any(strcmp(nm, reject)); cand(i) = []; end
    end
    nam_mat = fullfile(session_dir, cand(1).name);
end
data_file = matfile(nam_mat);
Ysiz = data_file.Ysiz; d1 = Ysiz(1); d2 = Ysiz(2); T = Ysiz(3);

% cnmfe_update_BG: Ybg = Y - neuron.A*neuron.C, then localBG(reshape(Ybg,2), ...)
Yin = reshape(double(data_file.Y), d1*d2, T) - neuron.A*neuron.C;
Yin = reshape(Yin, d1, d2, T);

spatial_ds_factor = 2;                                  % as cnmfe_update_BG sets it
bg_neuron_ratio   = 1.5;
thresh            = 10;
rr_arg = ceil(neuron.options.gSiz * bg_neuron_ratio);   % passed in un-halved
sn_arg = neuron.reshape(neuron.P.sn, 2);                % d1 x d2
active_px = [];

fprintf('[VERIFY] %d x %d x %d; ssub=%d, rr_arg=%d, thresh=%d\n', ...
    d1, d2, T, spatial_ds_factor, rr_arg, thresh);
fprintf('[VERIFY] input is %.1f GB; two outputs of the same size are kept\n', ...
    numel(Yin)*8/2^30);

%% ---- reference (pre-edit) ----
fprintf('[VERIFY] running local_background_orig (pre-edit)...\n');
t = tic;
[Yest_old, res_old] = local_background_orig(Yin, spatial_ds_factor, rr_arg, ...
    active_px, sn_arg, thresh);
s_old = toc(t);
fprintf('[VERIFY]   pre-edit:  %.1f s (%.1f min)\n', s_old, s_old/60);

%% ---- edited ----
fprintf('[VERIFY] running local_background (edited)...\n');
t = tic;
[Yest_new, res_new] = local_background(Yin, spatial_ds_factor, rr_arg, ...
    active_px, sn_arg, thresh);
s_new = toc(t);
fprintf('[VERIFY]   edited:    %.1f s (%.1f min)   speedup %.2fx\n', ...
    s_new, s_new/60, s_old/s_new);
clear Yin;

%% ---- compare everything returned ----
dabs = abs(Yest_old(:) - Yest_new(:));
dY   = max(dabs);
nd   = nnz(dabs);
ntot = numel(dabs);
clear dabs Yest_old Yest_new;

nbad = 0; dW = 0; nempty_mismatch = 0;
wo = res_old.weights; wn = res_new.weights;
same_shape = isequal(size(wo), size(wn));
for k = 1:numel(wo)
    a = wo{k}; b = wn{k};
    if isempty(a) && isempty(b); continue; end
    if isempty(a) ~= isempty(b)
        nempty_mismatch = nempty_mismatch + 1; continue;
    end
    if ~isequal(size(a), size(b)) || ~isequal(a(1,:), b(1,:))
        nbad = nbad + 1; continue;
    end
    dW = max(dW, max(abs(a(2,:) - b(2,:))));
end

db0 = max(abs(res_old.b0(:) - res_new.b0(:)));
ok_ssub = isequal(res_old.ssub, res_new.ssub);
ok_dims = isequal(res_old.dims, res_new.dims);

L = {};
L{end+1} = sprintf('local_background edit verification — %s', session_nm);
L{end+1} = datestr(now, 'yyyy-mm-dd HH:MM:SS');
L{end+1} = sprintf('%d x %d x %d; ssub=%d rr_arg=%d thresh=%d; active_px empty', ...
    d1, d2, T, spatial_ds_factor, rr_arg, thresh);
L{end+1} = '';
L{end+1} = sprintf('pre-edit  local_background_orig  %8.1f s  (%.1f min)', s_old, s_old/60);
L{end+1} = sprintf('edited    local_background       %8.1f s  (%.1f min)   speedup %.2fx', ...
    s_new, s_new/60, s_old/s_new);
L{end+1} = '';
L{end+1} = 'returned values compared:';
L{end+1} = sprintf('  Yest (upsampled, %d elements)   max|diff| %.3g   differing elements %d', ...
    ntot, dY, nd);
L{end+1} = sprintf('  results.weights cell            same shape %d   index mismatches %d   empty mismatches %d   max|coef diff| %.3g', ...
    same_shape, nbad, nempty_mismatch, dW);
L{end+1} = sprintf('  results.b0                      max|diff| %.3g', db0);
L{end+1} = sprintf('  results.ssub / results.dims     equal %d / %d', ok_ssub, ok_dims);
L{end+1} = '';
if dY == 0 && nd == 0 && nbad == 0 && nempty_mismatch == 0 && dW == 0 ...
        && db0 == 0 && same_shape && ok_ssub && ok_dims
    L{end+1} = 'VERDICT: BIT-IDENTICAL — the edited function returns exactly what the';
    L{end+1} = '         pre-edit function returned, in every field.';
else
    L{end+1} = 'VERDICT: *** DIFFERENCES FOUND — DO NOT DEPLOY ***';
end

txt = strjoin(L, newline);
fprintf('\n%s\n', txt);
out = fullfile(session_dir, 'localBG_verify_edit.txt');
fid = fopen(out, 'w'); fprintf(fid, '%s\n', txt); fclose(fid);
fprintf('[VERIFY] Report written to %s\n', out);
