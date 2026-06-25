%% CNMFe_Biane_headless.m
% Headless (no-GUI) version of CNMFe_Biane.m for automated processing.
%
% Called by the Python agent via:
%   matlab -batch "nam='<path>'; gSig=<n>; gSiz=<n>; min_corr=<n>; min_pnr=<n>; bd=<n>; run('CNMFe_Biane_headless.m')"
%
% All five hyperparameters must be set by the caller before running.
% Outputs are saved identically to the original pipeline PLUS Ybg.mat
% (needed for motion artifact detection).
%
% Changes from CNMFe_Biane.m are marked with: % [HEADLESS]

global  d1 d2 numFrame ssub tsub sframe num2read Fs neuron neuron_ds ...
neuron_full Ybg_weights; %#ok<NUSED>

% Suppress singular-matrix warnings from fit_gauss1/estimate_baseline_noise.
% These fire on degenerate (near-zero) traces and are expected for garbage
% candidates.  They flood the log without adding diagnostic value.
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:illConditionedMatrix');

%% select data and map it to the RAM
% [HEADLESS] nam is set by the caller — cnmfe_choose_data handles it when
% nam is pre-defined (skips the uigetfile dialog automatically)
cnmfe_choose_data;
% [HEADLESS] Allow caller to redirect all outputs to a different folder.
% Set dir_nm_override before calling this script to avoid overwriting
% existing session files (used by bootstrap_preagent.py).
if exist('dir_nm_override', 'var') && ~isempty(dir_nm_override); mkdir(dir_nm_override); dir_nm = dir_nm_override; end

%% create Source2D class object for storing results and parameters
Fs = 3.75;          % frame rate ***BE SURE TO CHANGE IF AVERAGING FRAMES***
ssub = 2;           % spatial downsampling factor
tsub = 2;           % temporal downsampling factor
% [HEADLESS] gSig and gSiz are set by the caller
neuron_full = Sources2D('d1',d1,'d2',d2, ...
    'ssub', ssub, 'tsub', tsub, ...
    'name',nam_mat,...
    'gSig', gSig,...
    'gSiz', gSiz);
neuron_full.Fs = Fs;

with_dendrites = false;
if with_dendrites
    neuron_full.options.search_method = 'dilate';
    neuron_full.options.bSiz = 20;
else
    neuron_full.options.search_method = 'ellipse';
    neuron_full.options.dist = 3;
end

merge_thr = [1e-1, 0.85, 0];
dmin = 1;

%% options for running deconvolution
neuron_full.options.deconv_options = struct('type', 'ar1', ...
    'method', 'thresholded', ...
    'optimize_pars', true, ...
    'optimize_b', true, ...
    'optimize_smin', false);

%% downsample data
sframe=1;
num2read= numFrame;

tic;
cnmfe_load_data;
fprintf('Time cost in downsampling data:     %.2f seconds\n', toc);

Y = neuron.reshape(Y, 1);

%% compute correlation image and peak-to-noise ratio image
% [HEADLESS] Run but do not display — we save Cn and pnr to disk so the
% Python agent can estimate min_corr/min_pnr for future sessions
cnmfe_show_corr_pnr;
close all;  % [HEADLESS] close the figures silently

%% initialization of A, C
debug_on = false;
save_avi = false;
patch_par = [1,1]*1;
K = [];

% [HEADLESS] min_corr, min_pnr, bd are set by the caller
min_pixel = 6^2;
neuron.updateParams('min_corr', min_corr, 'min_pnr', min_pnr, ...
    'min_pixel', min_pixel, 'bd', bd);
neuron.options.nk = 1;

tic;
[center, Cn, pnr] = neuron.initComponents_endoscope(Y, K, patch_par, debug_on, save_avi);
fprintf('Time cost in initializing neurons:     %.2f seconds\n', toc);

% [HEADLESS] skip figure display
neuron.orderROIs('snr');
neuron_init = neuron.copy();

% [HEADLESS] neuron.viewNeurons([], neuron.C_raw) call removed —
% no interactive deletion at this stage; all candidates are kept

%% iteratively update A, C and B
display_merge = false;  % [HEADLESS] no visual merge check
view_neurons = false;

spatial_ds_factor = 2;
thresh = 10;
bg_neuron_ratio = 1.5;

update_spatial_method = 'hals';
Nspatial = 5;

nC = size(neuron.C, 1);
maxIter = 3;
miter = 1;

while miter <= maxIter
    cnmfe_quick_merge;
    cnmfe_merge_neighbors;

    tic;
    cnmfe_update_BG;
    fprintf('Time cost in estimating the background:        %.2f seconds\n', toc);

    tic;
    for m=1:2
        try
            neuron.updateTemporal_endoscope(Ysignal);
        catch err_temporal
            fprintf('  [WARNING] updateTemporal_endoscope failed -- temporal components for degenerate neurons unchanged, all spatial footprints intact: %s\n', err_temporal.message);
        end
        cnmfe_quick_merge;
        neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
        neuron.trimSpatial(0.01, 3);
        neuron.compactSpatial();
        cnmfe_merge_neighbors;
        if isempty(merged_ROI)
            break;
        end
    end
    fprintf('Time cost in updating spatial & temporal components:     %.2f seconds\n', toc);

    if miter==1
        % [HEADLESS] use 'auto' seed selection instead of 'manual'
        seed_method = 'auto';  % [HEADLESS] was 'manual'
        [center_new, Cn_res, pnr_res] = neuron.pickNeurons(Ysignal - neuron.A*neuron.C, patch_par, seed_method, debug_on);
    end

    % [HEADLESS] miter==2 deletion step removed — neuron.viewNeurons call
    % skipped; all candidates proceed to the user's final review

    temp = size(neuron.C, 1);
    if or(nC==temp, miter==maxIter)
        break;
    else
        miter = miter+1;
        nC = temp;
    end
end

%% apply results to full resolution
if or(ssub>1, tsub>1)
    neuron_ds = neuron.copy();
    neuron = neuron_full.copy();
    cnmfe_full;
    neuron_full = neuron.copy();
end

% [HEADLESS] cnmfe_manual_merge removed — no interactive merge step

%% final update pass (mirrors the post-manual-merge update in original)
tic;
cnmfe_update_BG;
fprintf('Time cost in estimating the background:        %.2f seconds\n', toc);

tic;
for m=1:2
    try
        neuron.updateTemporal_endoscope(Ysignal);
    catch err_temporal
        fprintf('  [WARNING] updateTemporal_endoscope failed -- temporal components for degenerate neurons unchanged, all spatial footprints intact: %s\n', err_temporal.message);
    end
    cnmfe_quick_merge;
    neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
    neuron.trimSpatial(0.01, 3);
    neuron.compactSpatial();
    cnmfe_merge_neighbors;
end
fprintf('Time cost in updating spatial & temporal components:     %.2f seconds\n', toc);

b0 = mean(Y,2)-neuron.A*mean(neuron.C,2);
Ybg = bsxfun(@plus, Ybg, b0-mean(Ybg, 2));

% [HEADLESS] second cnmfe_manual_merge removed

try
    neuron.updateTemporal_endoscope(Ysignal);
catch err_temporal
    fprintf('  [WARNING] updateTemporal_endoscope failed (keeping previous temporal components): %s\n', err_temporal.message);
end
cnmfe_quick_merge;
neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
neuron.trimSpatial(0.01, 3);
neuron.compactSpatial();
cnmfe_merge_neighbors;

b0 = mean(Y,2)-neuron.A*mean(neuron.C,2);
Ybg = bsxfun(@plus, Ybg, b0-mean(Ybg, 2));

%% [HEADLESS] neuron.viewNeuronsVideo removed — user does this in final review

%% get contours
neuron.Coor = neuron.get_contours(0.8);

%% display and save ROI figure
figure('Visible','off');  % [HEADLESS] invisible figure
Cn = imresize(Cn, [d1, d2]);
plot_contours(neuron.A, Cn, 0.8, 0, [], neuron.Coor, 2);
colormap winter;
title('contours of estimated neurons (all candidates)');

%% save neuron images
dir_neurons = sprintf('%s%s%s_neurons%s', dir_nm, filesep, file_nm, filesep);
if exist(dir_neurons, 'dir')
    temp = cd();
    cd(dir_neurons);
    delete *;
    cd(temp);
else
    mkdir(dir_neurons);
end
neuron.viewNeurons([], neuron.C_raw, dir_neurons);
close all;  % [HEADLESS] close all figures after saving

%% extract dF/F
if size(neuron.A, 2) >= 2
    [neuron.C_df, neuron.Df] = extract_DF_F(Y, neuron.A, neuron.C_raw);
else
    % Single-neuron sessions: extract_DF_F assumes one column of A is a diffuse
    % background component (original CNMF) and zeros its dF/F.  In CNMF-E the
    % background lives in neuron.b/neuron.f, so the lone real neuron would be
    % wrongly flagged as background and zeroed.  Compute dF/F directly with no
    % background subtraction, matching extract_DF_F's defaults (df_prctile=50,
    % i.e. median baseline; no running window).
    A1  = full(neuron.A);
    nA1 = sqrt(sum(A1.^2));
    A1  = A1 / nA1;                % unit-energy footprint
    C1  = nA1 * neuron.C_raw;      % matching trace scaling
    Yf1 = A1' * Y;                 % projected fluorescence (no other components)
    neuron.Df   = median(Yf1, 2);  % baseline
    neuron.C_df = C1 / neuron.Df;
end

%% save all standard outputs (identical filenames to original pipeline)
cd(dir_nm);

dlmwrite('C_df.txt',  full(neuron.C_df),  'delimiter',' ')
dlmwrite('C_raw.txt', full(neuron.C_raw), 'delimiter',' ')
dlmwrite('C.txt',     full(neuron.C),     'delimiter',' ')
dlmwrite('Cnn.txt',   full(Cn),           'delimiter',' ')
dlmwrite('S.txt',     full(neuron.S),     'delimiter',' ')
dlmwrite('A.txt',     full(neuron.A),     'delimiter',' ')

overlapA = neuron.overlapA;
neuron.Coor = neuron.get_contours(0.8);
coor = neuron.Coor;

save('Coor.mat', 'coor');
% [HEADLESS] Skip neuron.mat in bootstrap mode — bootstrap_preagent.py only needs
% spatial_footprints.mat and C_raw.txt.  neuron.mat can be very large (100s of
% candidates) and causes a WD file-close error on large recordings.
if ~exist('dir_nm_override', 'var') || isempty(dir_nm_override)
    save('neuron.mat', 'neuron');
end
save('Cn.mat', 'Cn');
save('pnr.mat', 'pnr');          % [HEADLESS] saved for Python parameter estimation

% [HEADLESS] Save mean background per frame for motion artifact detection.
% Full Ybg is (pixels × T) — can exceed MATLAB v5 2GB limit on long recordings.
% Python only needs the per-frame mean to compute motion correlations.
mean_Ybg = mean(Ybg, 1);  % 1×T — mean pixel intensity per frame
save('Ybg_mean.mat', 'mean_Ybg');

%% save ROI figure
figure('Visible','off');
Cn = imresize(Cn, [d1, d2]);
plot_contours(neuron.A, Cn, 0.8, 0, [], neuron.Coor, 2);
colormap winter;
title('contours of estimated neurons (all candidates)');
saveas(gcf, 'ROIs_candidates.jpg');  % [HEADLESS] named _candidates to distinguish from final
close all;

%% spatial footprints for CellReg
clear spatial_footprints
number_of_cells = size(neuron.A, 2);
for n = 1:number_of_cells
    spatial_footprints(n,:,:) = reshape(neuron.A(:,n), d1, d2);
end
save('spatial_footprints.mat', 'spatial_footprints');

fprintf('\n[HEADLESS] Done. %d candidate neurons saved to %s\n', number_of_cells, dir_nm);
fprintf('[HEADLESS] Review package will be prepared by the Python agent.\n');
