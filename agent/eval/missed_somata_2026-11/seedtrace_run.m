%% seedtrace_run.m  -- SEEDTRACE diagnostic copy of CNMFe_Biane_headless.m lines 13-229
% agent/eval/missed_somata_2026-11, 2026-11-04.  Diagnostic only.
% Same computation as the production headless run; lines tagged SEEDTRACE
% only snapshot the component set (st_snap) or label the seeding stage for the
% shadow greedyROI_endoscope.m.  Deviations from the original, all output-free:
%   - cnmfe_choose_data is inlined WITHOUT its "save .dir.mat" (it would write
%     into the current folder); nam must be the movie .mat itself
%   - no dir_nm_override, no instrument block, and the script stops after the
%     last merge (line 229 of the original), before ANY session output is written
% Caller sets: nam, gSig, gSiz, min_corr, min_pnr, bd, st_out (existing folder).
% Writes only st_out/seedtrace_ds.mat and st_out/seedtrace.mat.

global  d1 d2 numFrame ssub tsub sframe num2read Fs neuron neuron_ds ...
neuron_full Ybg_weights; %#ok<NUSED>
global ST_SNAPS ST_SEEDS ST_STAGE  % SEEDTRACE
ST_SNAPS = {}; ST_SEEDS = {}; ST_STAGE = 'none';  % SEEDTRACE
assert(exist(st_out, 'dir') == 7, 'seedtrace: st_out folder missing');  % SEEDTRACE
st_t0 = tic;  % SEEDTRACE

% Suppress singular-matrix warnings from fit_gauss1/estimate_baseline_noise.
% These fire on degenerate (near-zero) traces and are expected for garbage
% candidates.  They flood the log without adding diagnostic value.
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:illConditionedMatrix');

%% select data and map it to the RAM
global data Ysiz d1 d2 numFrame;  % SEEDTRACE (cnmfe_choose_data inlined, no .dir.mat)
[dir_nm, file_nm, file_type] = fileparts(nam);  % SEEDTRACE (cnmfe_choose_data inlined)
nam_mat = [dir_nm, filesep, file_nm, '.mat'];  % SEEDTRACE (cnmfe_choose_data inlined)
assert(strcmpi(file_type, '.mat') && exist(nam_mat, 'file') == 2, 'seedtrace: movie .mat not found');  % SEEDTRACE
data = matfile(nam_mat);  % SEEDTRACE (cnmfe_choose_data inlined)
Ysiz = data.Ysiz;  % SEEDTRACE (cnmfe_choose_data inlined)
d1 = Ysiz(1);  % SEEDTRACE (cnmfe_choose_data inlined)
d2 = Ysiz(2);  % SEEDTRACE (cnmfe_choose_data inlined)
numFrame = Ysiz(3);  % SEEDTRACE (cnmfe_choose_data inlined)
fprintf('[SEEDTRACE] %s: %d x %d x %d\n', nam_mat, d1, d2, numFrame);  % SEEDTRACE

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
% [HEADLESS] Run but do not display -- we save Cn and pnr to disk so the
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
ST_STAGE = 'init';  % SEEDTRACE
[center, Cn, pnr] = neuron.initComponents_endoscope(Y, K, patch_par, debug_on, save_avi);
fprintf('Time cost in initializing neurons:     %.2f seconds\n', toc);
st_snap(neuron, 'init', [], []);  % SEEDTRACE

% [HEADLESS] skip figure display
neuron.orderROIs('snr');
st_snap(neuron, 'order', [], []);  % SEEDTRACE
neuron_init = neuron.copy();

% [HEADLESS] neuron.viewNeurons([], neuron.C_raw) call removed --
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
    st_snap(neuron, sprintf('L%d_qm0', miter), merged_ROI, newIDs);  % SEEDTRACE
    cnmfe_merge_neighbors;
    st_snap(neuron, sprintf('L%d_mn0', miter), merged_ROI, newIDs);  % SEEDTRACE

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
        st_snap(neuron, sprintf('L%d_%d_ut', miter, m), [], []);  % SEEDTRACE
        cnmfe_quick_merge;
        st_snap(neuron, sprintf('L%d_%d_qm', miter, m), merged_ROI, newIDs);  % SEEDTRACE
        neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
        st_snap(neuron, sprintf('L%d_%d_us', miter, m), [], []);  % SEEDTRACE
        neuron.trimSpatial(0.01, 3);
        neuron.compactSpatial();
        st_snap(neuron, sprintf('L%d_%d_tc', miter, m), [], []);  % SEEDTRACE
        cnmfe_merge_neighbors;
        st_snap(neuron, sprintf('L%d_%d_mn', miter, m), merged_ROI, newIDs);  % SEEDTRACE
        if isempty(merged_ROI)
            break;
        end
    end
    fprintf('Time cost in updating spatial & temporal components:     %.2f seconds\n', toc);

    if miter==1
        % [HEADLESS] use 'auto' seed selection instead of 'manual'
        seed_method = 'auto';  % [HEADLESS] was 'manual'
        ST_STAGE = 'pick';  % SEEDTRACE
        [center_new, Cn_res, pnr_res] = neuron.pickNeurons(Ysignal - neuron.A*neuron.C, patch_par, seed_method, debug_on);
        st_snap(neuron, 'L1_pick', [], []);  % SEEDTRACE
    end

    % [HEADLESS] miter==2 deletion step removed -- neuron.viewNeurons call
    % skipped; all candidates proceed to the user's final review

    temp = size(neuron.C, 1);
    if or(nC==temp, miter==maxIter)
        break;
    else
        miter = miter+1;
        nC = temp;
    end
end
st_d1 = d1; st_d2 = d2; st_numFrame = numFrame;  % SEEDTRACE
save(fullfile(st_out, 'seedtrace_ds.mat'), 'ST_SNAPS', 'ST_SEEDS', 'center', 'center_new', 'st_d1', 'st_d2', 'st_numFrame', '-v7');  % SEEDTRACE
fprintf('[SEEDTRACE] downsampled stage saved after %.1f min\n', toc(st_t0)/60);  % SEEDTRACE

%% apply results to full resolution
if or(ssub>1, tsub>1)
    neuron_ds = neuron.copy();
    neuron = neuron_full.copy();
    cnmfe_full;
    st_snap(neuron, 'F_full', [], []);  % SEEDTRACE
    neuron_full = neuron.copy();
end

% [HEADLESS] cnmfe_manual_merge removed -- no interactive merge step

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
    st_snap(neuron, sprintf('P_%d_ut', m), [], []);  % SEEDTRACE
    cnmfe_quick_merge;
    st_snap(neuron, sprintf('P_%d_qm', m), merged_ROI, newIDs);  % SEEDTRACE
    neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
    st_snap(neuron, sprintf('P_%d_us', m), [], []);  % SEEDTRACE
    neuron.trimSpatial(0.01, 3);
    neuron.compactSpatial();
    st_snap(neuron, sprintf('P_%d_tc', m), [], []);  % SEEDTRACE
    cnmfe_merge_neighbors;
    st_snap(neuron, sprintf('P_%d_mn', m), merged_ROI, newIDs);  % SEEDTRACE
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
st_snap(neuron, 'Q_ut', [], []);  % SEEDTRACE
cnmfe_quick_merge;
st_snap(neuron, 'Q_qm', merged_ROI, newIDs);  % SEEDTRACE
neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
st_snap(neuron, 'Q_us', [], []);  % SEEDTRACE
neuron.trimSpatial(0.01, 3);
neuron.compactSpatial();
st_snap(neuron, 'Q_tc', [], []);  % SEEDTRACE
cnmfe_merge_neighbors;
st_snap(neuron, 'Q_mn', merged_ROI, newIDs);  % SEEDTRACE

st_final_A = sparse(neuron.A);  % SEEDTRACE
save(fullfile(st_out, 'seedtrace.mat'), 'ST_SNAPS', 'ST_SEEDS', 'center', 'center_new', 'st_final_A', 'st_d1', 'st_d2', 'st_numFrame', '-v7');  % SEEDTRACE
fprintf('[SEEDTRACE] Done. %d final components, %.1f min. Saved to %s\n', size(neuron.A, 2), toc(st_t0)/60, st_out);  % SEEDTRACE
