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

% Reset the motion-delete accumulator for this session. viewNeurons /
% viewNeuronsVideo append a neuron's footprint here whenever the reviewer marks
% it 'm' (motion delete); we match those back to the review candidates at save
% time to record which deletes were motion artifacts.
global MOTION_DELETE_FP; %#ok<GVMIS>
MOTION_DELETE_FP = [];

%% --- Setup paths ---
repo_root = fileparts(mfilename('fullpath'));   % self-locating: this file lives at the repo root
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
% cvx ships pre-R2013a shims (lib/narginchk_) that shadow MATLAB builtins and
% warn on every launch; cvx_startup skips them on modern MATLAB, genpath cannot.
cvx_dirs = strsplit(genpath(fullfile(repo_root, 'cvx')), pathsep);
cvx_dirs = cvx_dirs(~cellfun(@isempty, cvx_dirs));
addpath(strjoin(cvx_dirs(~endsWith(cvx_dirs, [filesep 'narginchk_'])), pathsep));
clear cvx_dirs;
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

%% ========================================================
%% Resumable review: decide up front whether to resume
%% ========================================================
% review_checkpoint.mat lets a killed MATLAB session (forced OS restart, crash,
% or an accidental window close) resume near where it stopped instead of losing
% the whole review. We ask the reviewer to resume-or-restart HERE, before the
% ~20-minute video reload + background reconstruction, so they never sit through
% that only to discover a checkpoint was (or wasn't) waiting. The heavy preamble
% below ALWAYS re-runs regardless of the choice: it reloads the video and,
% critically, rebuilds Ysignal from the ORIGINAL review set so re-estimation is
% identical to an uninterrupted run. Only the curation state (the mutated neuron,
% the motion tags, and which step we reached) is swapped in later, AFTER Ysignal
% is fixed (see "swap in checkpointed curation state" below). A checkpoint is
% trusted only when its fingerprint matches this exact candidate set; on any
% mismatch we ignore it and start fresh rather than apply stale decisions to the
% wrong neurons.
[~, session_nm] = fileparts(session_dir);
CHECKPOINT_SCHEMA  = 1;
checkpoint_file    = fullfile(session_dir, 'review_checkpoint.mat');
review_fingerprint = struct( ...
    'session_nm',     session_nm, ...
    'N_review',       N_review, ...
    'a_checksum',     full(sum(A_review_init(:))), ...
    'schema_version', CHECKPOINT_SCHEMA);

resume_stage = 0;   % 0 = fresh start; N > 0 = resume after completing stage N
do_resume    = false;
stage_labels = { 'initial neuron review', 'intermediate update', ...
                 'pre-video merge', 'video inspection', 'manual merge', ...
                 'final update', 'final review loop' };

if isfile(checkpoint_file)
    try
        ckvars  = who('-file', checkpoint_file);
        has_all = all(ismember({'fingerprint', 'schema_version', 'step', ...
                                'neuron', 'motion_delete_fp'}, ckvars));
        if has_all
            % Read only the light-weight bookkeeping vars here; the 50+ MB neuron
            % object is loaded later, and only if the reviewer chooses to resume.
            ckmeta = load(checkpoint_file, 'fingerprint', 'schema_version', 'step');
            valid  = ckmeta.schema_version == CHECKPOINT_SCHEMA ...
                && strcmp(ckmeta.fingerprint.session_nm, review_fingerprint.session_nm) ...
                && ckmeta.fingerprint.N_review == review_fingerprint.N_review ...
                && abs(ckmeta.fingerprint.a_checksum - review_fingerprint.a_checksum) ...
                       <= 1e-6 * max(1, abs(review_fingerprint.a_checksum));
        else
            valid = false;
        end
    catch
        valid = false;   % unreadable / foreign checkpoint -- start fresh
    end
    if valid
        st  = ckmeta.step;
        lbl = 'an earlier step';
        if st >= 1 && st <= numel(stage_labels); lbl = stage_labels{st}; end
        choice = questdlg( ...
            sprintf(['A saved checkpoint was found for this session (stopped ' ...
                     'after: %s).\n\nResume from there, or start the review over ' ...
                     'from the beginning?\n\nEither way the video data reloads ' ...
                     'first (~20 min); your choice is applied once it finishes.'], lbl), ...
            'Resume Review', 'Resume', 'Start over', 'Resume');
        if strcmp(choice, 'Start over')
            delete(checkpoint_file);
            fprintf('  Checkpoint discarded; starting the review from the beginning.\n');
        else
            do_resume    = true;
            resume_stage = st;
            fprintf(['  Will resume after "%s" (stage %d) once the data finishes ' ...
                     'reloading -- your curated set is safe.\n'], lbl, st);
        end
    else
        fprintf(['  [INFO] A review_checkpoint.mat exists but does not match this ' ...
                 'session/candidate set (or its schema changed); ignoring it and ' ...
                 'starting fresh.\n']);
    end
end

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

% Locate the .mat data file (session_nm was computed up front for the checkpoint fingerprint)
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
%% Resumable review: swap in checkpointed curation state
%% ========================================================
% The resume/start-over decision was already made up front (right after
% review_neuron.mat loaded). Now that Ysignal has been rebuilt from the ORIGINAL
% review set, restore the checkpointed curation state on top of it so that
% re-estimation from here on is identical to an uninterrupted run.
if do_resume
    ck = load(checkpoint_file, 'neuron', 'motion_delete_fp');
    neuron           = ck.neuron;
    MOTION_DELETE_FP = ck.motion_delete_fp;
    clear ck;
    % Keep the restored neuron's data path valid if the session folder moved
    % since the checkpoint was written (mirrors the load-time sync above).
    if ~isfile(neuron.options.name)
        neuron.options.name = nam_mat;
    end
    lbl = 'an earlier step';
    if resume_stage >= 1 && resume_stage <= numel(stage_labels)
        lbl = stage_labels{resume_stage};
    end
    fprintf(['  Resumed after "%s" (stage %d): %d neurons and %d motion tag(s) ' ...
             'restored.\n'], lbl, resume_stage, size(neuron.A, 2), ...
             size(MOTION_DELETE_FP, 2));
end

%% ========================================================
%% STEP 1: Final viewNeurons — delete anything that looks wrong
%% ========================================================
if resume_stage < 1
    fprintf('\n--- STEP 1: Final neuron inspection (spatial footprints + traces) ---\n');
    fprintf('Delete any remaining bad neurons: (d) to delete, or (m) if the neuron\n');
    fprintf('is a motion artifact (logged as a motion delete). Close when done.\n\n');

    neuron.orderROIs('mean');
    neuron.viewNeurons([], neuron.C_raw);

    n_after_step1 = size(neuron.C, 1);
    fprintf('Neurons remaining after your review: %d\n', n_after_step1);
    save_review_checkpoint(session_dir, review_fingerprint, 1);
end

%% ========================================================
%% STEP 1b: Optional intermediate update (reclaim signal from deleted neighbors)
%% ========================================================
% If you deleted a neuron that was spatially overlapping with another, the
% neighbor's trace is stale — signal that was "stolen" by the deleted neuron
% won't appear until temporal components are re-estimated.
% Run this before video inspection so you judge corrected traces.
if resume_stage < 2
    answer_update = questdlg( ...
        ['Run intermediate update to reclaim signal from deleted neighbors?' ...
         ' (recommended if you deleted any overlapping neurons; ~1-2 min)'], ...
        'Intermediate Update', 'Yes', 'Skip', 'Skip');
else
    answer_update = 'Skip';   % already past this stage on resume; do not re-run
end
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
if resume_stage < 2
    save_review_checkpoint(session_dir, review_fingerprint, 2);
end

%% ========================================================
%% STEP 1c: Optional manual merge before video inspection
%% ========================================================
if resume_stage < 3
    answer_merge1 = questdlg('Do you want to run manual merge before video inspection?', ...
        'Pre-Video Merge', 'Yes', 'Skip', 'Skip');
else
    answer_merge1 = 'Skip';   % already past this stage on resume; do not re-run
end
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
if resume_stage < 3
    save_review_checkpoint(session_dir, review_fingerprint, 3);
end

%% ========================================================
%% STEP 2: viewNeuronsVideo — catch motion artifacts (repeatable)
%% ========================================================
if resume_stage < 4
    video_done = false;
    while ~video_done
        fprintf('\n--- STEP 2: Video inspection (motion artifact check) ---\n');
        fprintf('Watch for spikes that coincide with global brain motion.\n');
        fprintf('Tag motion artifacts with (m) so they are logged as motion deletes.\n');
        fprintf('Close the window when done.\n\n');

        neuron.viewNeuronsVideo([], neuron.C_raw);

        fprintf('Neurons remaining: %d\n', size(neuron.C, 1));

        answer_video = questdlg('Do you want to do another pass through the video?', ...
            'Video Review', 'Yes, go again', 'No, continue', 'No, continue');
        video_done = ~strcmp(answer_video, 'Yes, go again');
    end
    save_review_checkpoint(session_dir, review_fingerprint, 4);
end

%% ========================================================
%% STEP 3: Optional manual merge
%% ========================================================
if resume_stage < 5
    answer = questdlg('Do you want to run manual merge (for split cells)?', ...
        'Manual Merge', 'Yes', 'Skip', 'Skip');
else
    answer = 'Skip';   % already past this stage on resume; do not re-run
end
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
if resume_stage < 5
    save_review_checkpoint(session_dir, review_fingerprint, 5);
end

%% ========================================================
%% STEP 4: Final spatial/temporal update on curated set
%% ========================================================
if resume_stage < 6
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
    save_review_checkpoint(session_dir, review_fingerprint, 6);
end

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
    if ~review_done
        % Per-iteration checkpoint: the reviewer lingers in this loop making
        % deletes/merges, so snapshot after each completed pass. 'Save now' skips
        % this (we are about to finalize and clear the checkpoint below).
        save_review_checkpoint(session_dir, review_fingerprint, 7);
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

% --- Motion-delete labels ---
% Match the footprints the reviewer tagged 'm' (accumulated in MOTION_DELETE_FP by
% viewNeurons/viewNeuronsVideo) back to the review candidates by cosine similarity,
% the same way keep labels are matched above. motion_delete(i)=1 marks candidate i
% as deleted specifically for a motion artifact -- a SUBSET of the deletes; the
% keep/delete labels above are unaffected. Saved as an extra variable in labels.mat
% (train_classifier.py reads only 'labels' and ignores this). Note: a tagged
% footprint may have drifted slightly from its original candidate if updates ran
% before the tag, so an unmatched tag (corr < 0.60) is silently skipped rather than
% mislabelled.
motion_delete  = zeros(N_review, 1);
n_motion_marks = size(MOTION_DELETE_FP, 2);
if n_motion_marks > 0
    nmf   = sqrt(sum(MOTION_DELETE_FP.^2, 1)) + 1e-12;
    ncr   = sqrt(sum(A_review_init.^2,    1)) + 1e-12;
    mcorr = bsxfun(@rdivide, A_review_init, ncr)' * ...
            bsxfun(@rdivide, MOTION_DELETE_FP, nmf);   % N_review × n_marks
    for jm = 1:n_motion_marks
        [best_m, bm] = max(mcorr(:, jm));
        if best_m > 0.60
            motion_delete(bm) = 1;
        end
    end
end
% A motion delete is still a delete: never let it collide with a keep label.
motion_delete(labels == 1) = 0;
fprintf('  Motion deletes: %d tagged, %d matched to review candidates.\n', ...
    n_motion_marks, sum(motion_delete));

save(fullfile(session_dir, 'labels.mat'), 'labels', 'motion_delete', '-v7');
fprintf('  labels.mat saved.\n');
clear A_review_init A_final corr_mat nr nf used N_final motion_delete mcorr nmf ncr;
MOTION_DELETE_FP = [];

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

% Review finished and every output written -- remove the checkpoint so this
% completed session can never resume into a stale mid-review state.
if exist('checkpoint_file', 'var') && isfile(checkpoint_file)
    delete(checkpoint_file);
    fprintf('  Checkpoint cleared (session complete).\n');
end

fprintf('\n=== Done. %d final neurons saved to %s ===\n', number_of_cells, session_dir);
fprintf('You can now close MATLAB or process the next session.\n');


% -----------------------------------------------------------------------
% Local functions
% -----------------------------------------------------------------------

function save_review_checkpoint(session_dir, review_fingerprint, stage)
% Atomically write the resumable-review checkpoint at a step boundary. Reads the
% live curation state (neuron + motion tags) from the globals set by this script.
% Writing to a .tmp then renaming means a crash mid-write cannot corrupt an
% existing good checkpoint. The file is named review_checkpoint.mat so no
% downstream scanner (watcher.py / ingest_returns / train_classifier) mistakes it
% for a finished session. A checkpoint save must never abort the review, so any
% failure is caught and reported rather than raised.
    global neuron MOTION_DELETE_FP; %#ok<GVMIS>
    ckpt = struct();
    ckpt.neuron           = neuron;
    ckpt.motion_delete_fp = MOTION_DELETE_FP;
    ckpt.step             = stage;
    ckpt.fingerprint      = review_fingerprint;
    ckpt.schema_version   = review_fingerprint.schema_version;
    tmp_file  = fullfile(session_dir, 'review_checkpoint.mat.tmp');
    dest_file = fullfile(session_dir, 'review_checkpoint.mat');
    try
        save(tmp_file, '-struct', 'ckpt', '-v7');
        [ok, msg] = movefile(tmp_file, dest_file, 'f');
        if ~ok; error('movefile failed: %s', msg); end
        fprintf('  [checkpoint] saved after stage %d (%d neurons).\n', stage, size(neuron.A, 2));
    catch err_ckpt
        fprintf(2, '  [checkpoint] WARNING: could not save (%s); review continues.\n', err_ckpt.message);
        if isfile(tmp_file); delete(tmp_file); end
    end
end
