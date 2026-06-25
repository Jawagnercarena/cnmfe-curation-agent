%% CNMFe_final_save.m
% Run this after the Python agent has prepared your review package and you
% have finished your final inspection in MATLAB (viewNeurons + viewNeuronsVideo).
%
% This script:
%   1. Loads the agent's pre-curated neuron set (review_neuron.mat)
%   2. Lets you do a final viewNeurons pass (delete anything you want)
%   3. Lets you do viewNeuronsVideo (catch motion artifacts)
%   4. Runs one spatial/temporal update pass on your curated set
%   5. Saves all standard outputs (overwriting the headless candidate files)
%
% HOW TO RUN:
%   The agent creates a session-specific launcher script called
%   'run_final_review.m' inside the session folder. Open MATLAB, navigate
%   to the session folder, and run that script. It sets the session path
%   and calls this file automatically.
%
%   Or run manually:
%     session_dir = 'path\to\Area\Task\SessionName';
%     CNMFe_final_save     % with the CNMFe repo on your MATLAB path

global d1 d2 numFrame ssub tsub sframe num2read Fs neuron neuron_ds ...
    neuron_full Ybg_weights; %#ok<NUSED>

% Suppress singular-matrix warnings from fit_gauss1/estimate_baseline_noise.
% These fire on degenerate traces and are expected; they flood the log without
% adding diagnostic value.
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:illConditionedMatrix');

%% --- Setup paths ---
repo_root = fileparts(mfilename('fullpath'));   % self-locating: this file lives at the repo root
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
addpath(genpath(fullfile(repo_root, 'cvx')));
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error(['session_dir is not set. Either run the session-specific ' ...
           'run_final_review.m, or set session_dir manually before running this script.']);
end

%% --- Load review_neuron.mat prepared by the agent ---
review_file = fullfile(session_dir, 'review_neuron.mat');
if ~isfile(review_file)
    error('review_neuron.mat not found in %s\nHas the Python agent finished processing this session?', session_dir);
end

fprintf('Loading agent pre-curated neurons from review_neuron.mat ...\n');
data = load(review_file);
neuron = data.neuron;
Cn    = data.Cn;

fprintf('  %d neurons in the review set (agent already removed confident rejects).\n', size(neuron.C, 1));

% Snapshot initial review set for label extraction at the end of the session.
% A_review_init holds the spatial footprints of all candidates the user reviewed,
% so we can match them against the final curated set to derive keep/delete labels.
A_review_init = full(neuron.A);   % pixels × N_review
N_review       = size(A_review_init, 2);

% Restore key globals from neuron object
d1 = neuron.options.d1;
d2 = neuron.options.d2;
Fs = neuron.Fs;
ssub = neuron.options.ssub;
% After cnmfe_full, neuron.C has numFrame (full temporal resolution) frames.
% Load data with tsub=1 so Y matches — using options.tsub would halve the frames.
tsub = 1;

%% --- Reload raw video data (needed for viewNeuronsVideo and updates) ---
fprintf('\nReloading raw video data (this may take a moment)...\n');

% Locate the .mat data file
[~, session_nm] = fileparts(session_dir);
nam_mat = fullfile(session_dir, [session_nm, '.mat']);
if ~isfile(nam_mat)
    % try .tif -> .mat conversion name pattern
    nam_mat_files = dir(fullfile(session_dir, '*.mat'));
    % exclude output .mat files (neuron.mat, Cn.mat, etc.)
    exclude = {'neuron','Cn','Coor','pnr','Ybg','Ybg_mean','spatial_footprints','review_neuron'};
    for i = numel(nam_mat_files):-1:1
        [~, nm] = fileparts(nam_mat_files(i).name);
        if any(strcmp(nm, exclude)); nam_mat_files(i) = []; end
    end
    if isempty(nam_mat_files)
        error('Could not find the raw data .mat file in %s', session_dir);
    end
    nam_mat = fullfile(session_dir, nam_mat_files(1).name);
end

data_file = matfile(nam_mat);
Ysiz = data_file.Ysiz;
d1 = Ysiz(1); d2 = Ysiz(2); numFrame = Ysiz(3);

% After cnmfe_full, neuron.A is at FULL spatial resolution (d1×d2).
% We must load Y at full resolution too (ssub=1, tsub=1).
% cnmfe_load_data only takes the no-downsampling path when ssub==1 AND tsub==1;
% that path reads data.Y directly — so we need 'data' in the workspace.
% We also save/restore neuron because cnmfe_load_data overwrites it.
data         = data_file;       % cnmfe_load_data uses 'data.Y' in the fast path
ssub         = 1;
tsub         = 1;
neuron_full  = neuron.copy();   % cnmfe_load_data sets neuron=neuron_full in fast path
neuron_saved = neuron.copy();   % preserve curated set

sframe = 1;
num2read = numFrame;
nam = nam_mat;
fprintf('Loading full-resolution data (%d x %d x %d frames)...\n', d1, d2, numFrame);
cnmfe_load_data;   % Y = double array (d1 x d2 x numFrame), reshaped below

neuron = neuron_saved;   % restore curated set (cnmfe_load_data overwrote it)
clear neuron_saved;

% Sync neuron.options.name to the actual data file location.
% After a data migration (e.g. BLA move to BLA/ subfolder), the path stored
% inside review_neuron.mat may be stale.  nam_mat was derived from session_dir
% above, which is always correct.
if ~isfile(neuron.options.name)
    neuron.options.name = nam_mat;
    fprintf('  [INFO] Updated neuron.options.name to: %s\n', nam_mat);
end

Y = neuron.reshape(Y, 1);

% Re-estimate background.
% Fast path: if the watcher pre-computed Ybg_weights.mat, replay with reconstructBG
% (~seconds).  Slow path: run localBG from scratch via cnmfe_update_BG (~10 min).
ybg_cache = fullfile(session_dir, 'Ybg_weights.mat');
if isfile(ybg_cache)
    fprintf('\nLoading pre-computed background weights (fast path)...\n');
    wdata   = load(ybg_cache);
    weights = wdata.Ybg_weights;
    clear wdata;
    tic;
    % Inline reconstructBG without parfor.
    % neuron.reconstructBG uses parfor over ~65k pixels; MATLAB tries to serialize Y
    % (~3.5 GB) to each of 24 workers, causing OOM during deserialization.
    % This for-loop version is identical in result; ~30-60 s single-threaded.
    Yres    = neuron.reshape(Y - neuron.A * neuron.C, 2);   % [d1 × d2 × T]
    [d1f, d2f, T_bg] = size(Yres);
    dims    = weights.dims;                                   % downsampled size, e.g. [256 256]
    b0_bg   = mean(Yres, 3);                                 % [d1 × d2] per-pixel temporal mean
    Yds     = imresize(bsxfun(@minus, Yres, b0_bg), dims);   % demean + downsample → [d1s × d2s × T]
    clear Yres;
    Yds     = reshape(Yds, [], T_bg);                        % [nPix_ds × T]
    nPix_ds = size(Yds, 1);
    Ybg_ds  = zeros(nPix_ds, T_bg);
    for m = 1:nPix_ds
        w = weights.weights{m};
        Ybg_ds(m, :) = w(2,:) * Yds(w(1,:), :);
    end
    clear Yds weights;
    Ybg_ds  = reshape(Ybg_ds, [dims, T_bg]);
    Ybg     = bsxfun(@plus, imresize(Ybg_ds, [d1f, d2f]), b0_bg);  % upsample + add mean
    clear Ybg_ds b0_bg;
    Ybg     = neuron.reshape(Ybg, 1);                        % [d1*d2 × T]
    Ysignal = Y - Ybg;
    fprintf('Background reconstructed (%.1f s).\n', toc);
else
    fprintf('\nEstimating background (no pre-computed weights — takes ~10 min)...\n');
    spatial_ds_factor = 2;
    thresh = 10;
    bg_neuron_ratio = 1.5;
    cnmfe_update_BG;
end

% Parameters needed by cnmfe_quick_merge and cnmfe_merge_neighbors
merge_thr     = [1e-1, 0.85, 0];
dmin          = 1;
display_merge = false;
view_neurons  = false;
update_spatial_method = 'hals';
Nspatial      = 5;

%% ========================================================
%% STEP 1: Final viewNeurons — delete anything that looks wrong
%% ========================================================
fprintf('\n--- STEP 1: Final neuron inspection (spatial footprints + traces) ---\n');
fprintf('Delete any remaining bad neurons. Close the window when done.\n\n');

neuron.orderROIs('mean');
neuron.viewNeurons([], neuron.C_raw);

n_after_step1 = size(neuron.C, 1);
fprintf('Neurons remaining after your review: %d\n', n_after_step1);

%% ========================================================
%% STEP 1b: Optional intermediate update (reclaim signal from deleted neighbors)
%% ========================================================
% If you deleted a neuron that was spatially overlapping with another, the
% neighbor's trace is stale — signal that was "stolen" by the deleted neuron
% won't appear until temporal components are re-estimated.
% Run this before video inspection so you judge corrected traces.
answer_update = questdlg( ...
    ['Run intermediate update to reclaim signal from deleted neighbors?' ...
     ' (recommended if you deleted any overlapping neurons; ~1-2 min)'], ...
    'Intermediate Update', 'Yes', 'Skip', 'Skip');
if strcmp(answer_update, 'Yes')
    fprintf('\n--- Intermediate update (temporal + spatial, no BG re-estimation) ---\n');
    tic;
    for m = 1:2
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
    fprintf('Intermediate update done (%.1f s). Neurons: %d\n', toc, size(neuron.C, 1));
end

%% ========================================================
%% STEP 1c: Optional manual merge before video inspection
%% ========================================================
answer_merge1 = questdlg('Do you want to run manual merge before video inspection?', ...
    'Pre-Video Merge', 'Yes', 'Skip', 'Skip');
if strcmp(answer_merge1, 'Yes')
    cnmfe_manual_merge;
    waitfor(fig_merge);
    % Log merge decisions for pairwise merge classifier training
    if exist('ind_del', 'var') && any(ind_del)
        merge_log_file = fullfile(session_dir, sprintf('merge_log_%s.mat', datestr(now, 'HHMMSS')));
        save(merge_log_file, 'neuron_bk', 'ind_del');
        fprintf('Merge log saved: %d neurons merged (%s)\n', sum(ind_del), merge_log_file);
        % Re-estimate traces so video inspection sees the corrected merged traces.
        fprintf('\n--- Post-merge update (re-estimating temporal & spatial components) ---\n');
        tic;
        for m = 1:2
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
        fprintf('Post-merge update done (%.1f s). Neurons: %d\n', toc, size(neuron.C, 1));
    end
end

%% ========================================================
%% STEP 2: viewNeuronsVideo — catch motion artifacts (repeatable)
%% ========================================================
video_done = false;
while ~video_done
    fprintf('\n--- STEP 2: Video inspection (motion artifact check) ---\n');
    fprintf('Watch for spikes that coincide with global brain motion.\n');
    fprintf('Close the window when done.\n\n');

    neuron.viewNeuronsVideo([], neuron.C_raw);

    fprintf('Neurons remaining: %d\n', size(neuron.C, 1));

    answer_video = questdlg('Do you want to do another pass through the video?', ...
        'Video Review', 'Yes, go again', 'No, continue', 'No, continue');
    video_done = ~strcmp(answer_video, 'Yes, go again');
end

%% ========================================================
%% STEP 3: Optional manual merge
%% ========================================================
answer = questdlg('Do you want to run manual merge (for split cells)?', ...
    'Manual Merge', 'Yes', 'Skip', 'Skip');
if strcmp(answer, 'Yes')
    cnmfe_manual_merge;
    waitfor(fig_merge);   % block until FINISH button closes the figure
    % Log merge decisions for pairwise merge classifier training
    if exist('ind_del', 'var') && any(ind_del)
        merge_log_file = fullfile(session_dir, sprintf('merge_log_%s.mat', datestr(now, 'HHMMSS')));
        save(merge_log_file, 'neuron_bk', 'ind_del');
        fprintf('Merge log saved: %d neurons merged (%s)\n', sum(ind_del), merge_log_file);
    end
end

%% ========================================================
%% STEP 4: Final spatial/temporal update on curated set
%% ========================================================
fprintf('\n--- Updating spatial & temporal components on curated set ---\n');

tic;
for m = 1:2
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
fprintf('Update done (%.1f s). Final neuron count: %d\n', toc, size(neuron.C, 1));

%% ========================================================
%% STEP 4b: Final review loop — keep deleting / merging until satisfied
%% ========================================================
review_done = false;
while ~review_done
    answer_final = questdlg( ...
        sprintf('Neurons remaining: %d.  Make any more changes before saving?', size(neuron.C, 1)), ...
        'Final Check', 'Delete neurons', 'Merge neurons', 'Save now', 'Save now');

    if strcmp(answer_final, 'Delete neurons')
        neuron.viewNeurons([], neuron.C_raw);
        fprintf('Neurons remaining after deletion: %d\n', size(neuron.C, 1));
        fprintf('Updating components...\n'); tic;
        for m = 1:2
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
        fprintf('Update done (%.1f s). Neurons: %d\n', toc, size(neuron.C, 1));
        % Offer video re-check after deletion so the user can catch any
        % motion artifacts introduced by the updated traces.
        answer_vid = questdlg('Run video inspection on updated set?', ...
            'Video Check', 'Yes', 'Skip', 'Skip');
        if strcmp(answer_vid, 'Yes')
            vid_done = false;
            while ~vid_done
                fprintf('\n--- Video inspection (post-deletion) ---\n');
                neuron.viewNeuronsVideo([], neuron.C_raw);
                fprintf('Neurons remaining: %d\n', size(neuron.C, 1));
                ans_vid2 = questdlg('Another video pass?', 'Video Review', ...
                    'Yes, go again', 'No, continue', 'No, continue');
                vid_done = ~strcmp(ans_vid2, 'Yes, go again');
            end
        end

    elseif strcmp(answer_final, 'Merge neurons')
        cnmfe_manual_merge;
        waitfor(fig_merge);
        if exist('ind_del', 'var') && any(ind_del)
            merge_log_file = fullfile(session_dir, sprintf('merge_log_%s.mat', datestr(now, 'HHMMSS')));
            save(merge_log_file, 'neuron_bk', 'ind_del');
            fprintf('Merge log saved: %d neurons merged (%s)\n', sum(ind_del), merge_log_file);
            fprintf('Updating components after merge...\n'); tic;
            for m = 1:2
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
            fprintf('Update done (%.1f s). Neurons: %d\n', toc, size(neuron.C, 1));
            % Offer video re-check after merge.
            answer_vid = questdlg('Run video inspection on updated set?', ...
                'Video Check', 'Yes', 'Skip', 'Skip');
            if strcmp(answer_vid, 'Yes')
                vid_done = false;
                while ~vid_done
                    fprintf('\n--- Video inspection (post-merge) ---\n');
                    neuron.viewNeuronsVideo([], neuron.C_raw);
                    fprintf('Neurons remaining: %d\n', size(neuron.C, 1));
                    ans_vid2 = questdlg('Another video pass?', 'Video Review', ...
                        'Yes, go again', 'No, continue', 'No, continue');
                    vid_done = ~strcmp(ans_vid2, 'Yes, go again');
                end
            end
        end

    else  % 'Save now' or dialog closed
        review_done = true;
    end
end

%% --- Save keep/delete labels for classifier training ---
% Compare A_review_init (candidates seen by the user) to the current curated
% neuron.A using cosine similarity.  Any review candidate that strongly matches
% a final neuron is labelled keep=1; unmatched candidates are delete=0.
% train_classifier.py reads candidate_features.npz + labels.mat to train.
fprintf('\nSaving keep/delete labels for classifier training...\n');
A_final   = full(neuron.A);   % pixels × N_kept (current curated set)
N_final   = size(A_final, 2);
nr        = sqrt(sum(A_review_init.^2, 1)) + 1e-12;
nf        = sqrt(sum(A_final.^2,       1)) + 1e-12;
corr_mat  = bsxfun(@rdivide, A_review_init, nr)' * ...
            bsxfun(@rdivide, A_final, nf);             % N_review × N_final
% One-to-one greedy matching: each final neuron claims its best
% available review candidate above the threshold.  Without this,
% a deleted neuron overlapping a kept neighbor gets a false keep=1 label.
labels    = zeros(N_review, 1);
used      = false(N_review, 1);
for jf = 1:N_final
    col              = corr_mat(:, jf);
    col(used)        = -Inf;
    [best_corr, bi]  = max(col);
    if best_corr > 0.60
        labels(bi) = 1;
        used(bi)   = true;
    end
end
fprintf('  Labels: %d kept, %d deleted (from %d review candidates)\n', ...
    sum(labels), sum(labels == 0), N_review);
save(fullfile(session_dir, 'labels.mat'), 'labels', '-v7');
fprintf('  labels.mat saved.\n');
clear A_review_init A_final corr_mat nr nf used N_final;

b0 = mean(Y,2) - neuron.A*mean(neuron.C,2);
Ybg = bsxfun(@plus, Ybg, b0 - mean(Ybg, 2));

%% ========================================================
%% STEP 5: Extract dF/F and save all outputs
%% ========================================================
fprintf('\n--- Saving final outputs to %s ---\n', session_dir);
cd(session_dir);

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

dlmwrite('C_df.txt',  full(neuron.C_df),  'delimiter',' ')
dlmwrite('C_raw.txt', full(neuron.C_raw), 'delimiter',' ')
dlmwrite('C.txt',     full(neuron.C),     'delimiter',' ')
dlmwrite('Cnn.txt',   full(Cn),           'delimiter',' ')
dlmwrite('S.txt',     full(neuron.S),     'delimiter',' ')
dlmwrite('A.txt',     full(neuron.A),     'delimiter',' ')

clear spatial_footprints
number_of_cells = size(neuron.A, 2);
for n = 1:number_of_cells
    spatial_footprints(n,:,:) = reshape(neuron.A(:,n), d1, d2);
end

neuron.Coor = neuron.get_contours(0.8);
coor = neuron.Coor;

save('Coor.mat', 'coor');
save('neuron.mat', 'neuron');
save('Cn.mat', 'Cn');
save('spatial_footprints.mat', 'spatial_footprints');

% Save final ROI figure
figure;
Cn_full = imresize(Cn, [d1, d2]);
plot_contours(neuron.A, Cn_full, 0.8, 0, [], neuron.Coor, 2);
colormap winter;
title(sprintf('Final curated neurons (n=%d)', number_of_cells));
saveas(gcf, 'ROIs.jpg');
close(gcf);

% Save individual neuron images (overwrites candidate set)
dir_neurons = sprintf('%s%s%s_neurons%s', session_dir, filesep, session_nm, filesep);
if exist(dir_neurons, 'dir')
    temp_cd = cd();
    cd(dir_neurons); delete *; cd(temp_cd);
else
    mkdir(dir_neurons);
end
ws = warning('off', 'all');   % suppress R2025b UI-in-figure saveas warnings
neuron.viewNeurons([], neuron.C_raw, dir_neurons);
warning(ws);
close(gcf);

fprintf('\n=== Done. %d final neurons saved to %s ===\n', number_of_cells, session_dir);
fprintf('You can now close MATLAB or process the next session.\n');
