%% CNMFe_precompute_BG.m
% Pre-computes and caches the background spatial weights (Ybg_weights) so
% that run_final_review.m can skip the slow (~10 min) localBG step and use
% the fast reconstructBG path instead.
%
% Called by the Python watcher after the curator finishes.
% session_dir must be set by the caller.
%
% Output: Ybg_weights.mat in the session folder (~100-200 MB).
%   Contains the LLE ring-model weights struct from localBG.
%   run_final_review.m loads this and calls neuron.reconstructBG(Y)
%   which replays the weights as fast matrix multiplications.

global d1 d2 numFrame ssub tsub sframe num2read Fs neuron neuron_ds ...
    neuron_full Ybg_weights; %#ok<NUSED>

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
addpath(genpath(fullfile(repo_root, 'cvx')));
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('session_dir must be set before running CNMFe_precompute_BG.m');
end

ybg_out = fullfile(session_dir, 'Ybg_weights.mat');
if isfile(ybg_out)
    fprintf('[PRECOMPUTE_BG] Ybg_weights.mat already exists — nothing to do.\n');
    return;
end

%% Load saved neuron object
fprintf('[PRECOMPUTE_BG] Loading neuron.mat...\n');
ndata  = load(fullfile(session_dir, 'neuron.mat'));
neuron = ndata.neuron;
Fs     = neuron.Fs;

%% Locate raw data .mat file (same logic as CNMFe_final_save.m)
[~, session_nm] = fileparts(session_dir);
nam_mat = fullfile(session_dir, [session_nm, '.mat']);
if ~isfile(nam_mat)
    nam_mat_files = dir(fullfile(session_dir, '*.mat'));
    exclude = {'neuron','Cn','Coor','pnr','Ybg','Ybg_mean', ...
               'spatial_footprints','review_neuron','Ybg_weights'};
    for i = numel(nam_mat_files):-1:1
        [~, nm] = fileparts(nam_mat_files(i).name);
        if any(strcmp(nm, exclude)); nam_mat_files(i) = []; end
    end
    if isempty(nam_mat_files)
        error('[PRECOMPUTE_BG] Could not find raw data .mat in %s', session_dir);
    end
    nam_mat = fullfile(session_dir, nam_mat_files(1).name);
end

%% Load Y at full resolution (no spatial or temporal downsampling)
% neuron.A is at full resolution after cnmfe_full — Y must match.
data_file = matfile(nam_mat);
Ysiz      = data_file.Ysiz;
d1 = Ysiz(1); d2 = Ysiz(2); numFrame = Ysiz(3);

fprintf('[PRECOMPUTE_BG] Loading full-resolution data (%d x %d x %d frames)...\n', ...
    d1, d2, numFrame);
data         = data_file;     % cnmfe_load_data uses 'data.Y' when ssub==1, tsub==1
ssub         = 1;
tsub         = 1;
neuron_full  = neuron.copy(); % cnmfe_load_data sets neuron=neuron_full in fast path
neuron_saved = neuron.copy(); % preserve loaded neuron

sframe   = 1;
num2read = numFrame;
nam      = nam_mat;
cnmfe_load_data;   % loads Y at full resolution

neuron = neuron_saved;   % restore (cnmfe_load_data overwrote it)
clear neuron_saved;

Y = neuron.reshape(Y, 1);   % (d1*d2 x numFrame)

%% Estimate background — the slow step (localBG, ~10 min)
fprintf('[PRECOMPUTE_BG] Estimating background spatial weights (this takes ~10 min)...\n');
spatial_ds_factor = 2;
thresh            = 10;
bg_neuron_ratio   = 1.5;
tic;
cnmfe_update_BG;
fprintf('[PRECOMPUTE_BG] Done (%.1f s).\n', toc);

%% Save only Ybg_weights — the LLE weight struct (~100-200 MB)
% Ybg itself is ~27 GB and cannot be saved practicably.
% Ybg_weights contains the ring-model coefficients; reconstructBG uses them
% to recompute Ybg instantly from any new residual.
save(ybg_out, 'Ybg_weights');
fprintf('[PRECOMPUTE_BG] Saved Ybg_weights.mat to %s\n', session_dir);
