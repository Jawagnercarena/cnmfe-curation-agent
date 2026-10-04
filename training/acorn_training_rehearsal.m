function results = acorn_training_rehearsal(session_dir, key, opts)
% acorn_training_rehearsal -- run the REAL review tool, then score its labels.
%
%   results = acorn_training_rehearsal(session_dir, key, opts)
%
% Stage 4 of the training program.  Runs CNMFe_final_save on this training
% copy exactly as run_final_review.m does (in the base workspace, with
% session_dir set there), so the trainee goes through the full workflow:
% video reload, background rebuild, viewNeurons, updates, merges,
% viewNeuronsVideo, save.  The labels.mat the tool derives from the final
% footprints is then scored against the key with the same scorer the drill
% uses, and the usual report / progress row are written (mode 'rehearsal').
%
% The tool writes neuron.mat, labels.mat, traces, ROIs.jpg and
% <session>_neurons/ INTO THIS FOLDER.  That is intended: the folder is a
% sandbox (TRAINING_SESSION.txt marks it and the central ingest refuses it).
% Never copy it into inbox/.
%
% opts: trainee, stage (default 4), out_dir, progress_csv, clear_base (default
% true: clear the big video arrays from the base workspace afterwards),
% headless (report only).

if nargin < 3 || isempty(opts); opts = struct(); end
o = struct();
o.trainee = getopt(opts, 'trainee', 'trainee');
o.stage = getopt(opts, 'stage', 4);
o.out_dir = getopt(opts, 'out_dir', fullfile(session_dir, 'training_results'));
o.progress_csv = getopt(opts, 'progress_csv', '');
o.clear_base = logical(getopt(opts, 'clear_base', true));
o.headless = logical(getopt(opts, 'headless', false));
o.exclude_contested = logical(getopt(opts, 'exclude_contested', true));
session_dir = strip_sep(char(session_dir));
repo_root = fileparts(fileparts(mfilename('fullpath')));
results = [];

% --- pre-flight ----------------------------------------------------------------------
[~, session_nm] = fileparts(session_dir);
nam_mat = fullfile(session_dir, [session_nm, '.mat']);
if ~isfile(nam_mat)
    error(['Raw video %s not found. The dress rehearsal needs the full bundle ' ...
           '(pushed without --no-video).'], nam_mat);
end
if isempty(which('CNMFe_final_save'))
    error('CNMFe_final_save.m is not on the MATLAB path. Run addpath(genpath(<repo_root>)) first.');
end
if ~isfile(fullfile(session_dir, 'Ybg_weights.mat'))
    fprintf(2, 'Ybg_weights.mat is missing: the tool will rebuild the background from scratch (slow path, ~10 min).\n');
end
try
    mf = matfile(nam_mat);
    Ysiz = double(mf.Ysiz);
    need = prod(Ysiz) * 8 * 3;
    fprintf('Video %d x %d x %d frames; the real tool holds it as double (about %.0f GB with its working copies).\n', ...
        Ysiz(1), Ysiz(2), Ysiz(3), need / 1e9);
    try
        mem = memory;
        if mem.MaxPossibleArrayBytes < need
            fprintf(2, 'WARNING: this machine may not have enough memory for the full review (%.0f GB available).\n', ...
                mem.MaxPossibleArrayBytes / 1e9);
        end
    catch
    end
catch
end
if isfile(fullfile(session_dir, 'review_checkpoint.mat'))
    fprintf(['A review_checkpoint.mat exists here: the tool will ask whether to resume or start over ' ...
             '("Start over" deletes it). The trainer never touches the checkpoint.\n']);
end
fprintf('\nThe real review tool takes ~20 minutes to reload the video before the first prompt.\n');
fprintf('Work through it exactly as a real review. Your decisions become labels.mat in this folder and are scored afterwards.\n\n');

labels_path = fullfile(session_dir, 'labels.mat');
labels_before = NaN;
if isfile(labels_path)
    d = dir(labels_path);
    labels_before = d.datenum;
end
cwd0 = pwd;
started_at = now_str();
t0 = tic;

% --- run the real tool in the base workspace (exactly what run_final_review.m does) ---
assignin('base', 'session_dir', session_dir);
tool_error = '';
try
    evalin('base', 'CNMFe_final_save');
catch err
    tool_error = err.message;
    fprintf(2, '\nCNMFe_final_save stopped: %s\n', err.message);
end
cd(cwd0);
elapsed = toc(t0);
if o.clear_base
    try
        evalin('base', 'clear Y Ybg Ysignal Yres data Y0');
    catch
    end
end

% --- did the review reach the save step? ------------------------------------------------
ts = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
if ~exist(o.out_dir, 'dir'); mkdir(o.out_dir); end
ok = isfile(labels_path);
if ok && ~isnan(labels_before)
    d = dir(labels_path);
    ok = d.datenum > labels_before;
end
if ~ok
    note = fullfile(o.out_dir, sprintf('rehearsal_%s_%s_incomplete.txt', key.session, ts));
    fid = fopen(note, 'w');
    fprintf(fid, 'started: %s\nelapsed_sec: %.0f\ntool_error: %s\nno new labels.mat -- the review did not reach the save step; nothing scored\n', ...
        started_at, elapsed, tool_error);
    fclose(fid);
    fprintf(2, 'No new labels.mat was written -- the review did not reach the save step; nothing scored (%s).\n', note);
    return;
end

L = load(labels_path);
labels = double(L.labels(:));
if isfield(L, 'motion_delete'); motion = double(L.motion_delete(:)); else; motion = zeros(size(labels)); end
if numel(labels) ~= key.n_review
    error('labels.mat has %d entries but the key expects %d -- the candidate set changed; not scored.', ...
        numel(labels), key.n_review);
end
S = acorn_training_score(labels, motion, key, struct('exclude_contested', o.exclude_contested));

% --- results in the drill's shape so the report can draw panels --------------------------
rn = load(fullfile(session_dir, 'review_neuron.mat'));
neuron = rn.neuron;
Cn = []; if isfield(rn, 'Cn'); Cn = rn.Cn; end
clear rn;
ctx = acorn_training_context(neuron, Cn, key);
n = ctx.n;
results = struct();
results.schema_version = 1;
results.trainee = o.trainee;
results.area = key.area; results.task = key.task; results.session = key.session;
results.stage = o.stage;
results.mode = 'rehearsal';
results.drill_mode = 'rehearsal';
results.score_basis = 'last';
results.n_corrected = 0;
results.triage = [];
results.subset = 'all';
results.show_cn = false;
results.video_pass = true;
results.started_at = started_at;
results.finished_at = now_str();
results.ts = ts;
results.repo_hash = repo_hash(repo_root);
results.matlab_version = version;
results.n_review = n;
results.perm = ctx.perm;
results.review_col = ctx.perm;
results.shown_pos = (1:n)';
results.decision = int8(labels(ctx.perm));
results.motion = int8(motion(ctx.perm));
results.visited = true(n, 1);
results.n_views = ones(n, 1);
results.seconds = zeros(n, 1);
results.flagged = false(n, 1);
results.changed_after_feedback = false(n, 1);
results.decision_full = labels;
results.motion_full = motion;
results.decision_last_full = labels;
results.motion_last_full = motion;
results.subset_mask = true(n, 1);
results.score = S;
results.score2 = [];
results.score_final = S;
results.pass2 = [];
results.key_path = key.path;
results.tool_error = tool_error;
results.total_sec = elapsed;
result_file = fullfile(o.out_dir, sprintf('rehearsal_%s_%s.mat', key.session, ts));
save(result_file, 'results', 'key', 'labels', 'motion', '-v7');
results.result_file = result_file;

row = struct();
row.timestamp = results.finished_at; row.trainee = o.trainee;
row.area = key.area; row.task = key.task; row.session = key.session;
row.stage = o.stage; row.mode = 'rehearsal'; row.pass = 'final'; row.subset = 'all';
row.n_items = n; row.n_visited = n; row.n_scored = S.n_scored; row.n_contested = S.n_contested;
row.n_ref_keep_scored = S.n_ref_keep_scored; row.n_ref_delete_scored = S.n_ref_delete_scored;
row.agreement = S.agreement; row.kappa = S.kappa;
row.false_keep = S.false_keep; row.false_delete = S.false_delete;
row.motion_ref_n = S.motion_ref_n; row.motion_tagged_m = S.motion_tagged_m;
row.motion_deleted_any = S.motion_deleted_any; row.motion_false_tags = S.motion_false_tags;
row.n_flagged = 0; row.median_sec = NaN; row.total_sec = elapsed;
[~, nm, ext] = fileparts(result_file); row.result_file = [nm, ext];
row.repo_hash = results.repo_hash;
row.contested_with_ref = S.contested_with_ref;
row.contested_with_model = S.contested_with_model;
row.n_corrected = 0;
acorn_training_append_progress(fullfile(o.out_dir, 'progress.csv'), row);
if ~isempty(o.progress_csv); acorn_training_append_progress(o.progress_csv, row); end

fprintf('\n--- Dress rehearsal result (%s) ---\n', key.session);
fprintf('scored %d of %d (%d contested not counted); agreement %.3f, kappa %.3f; false keeps %d of %d, false deletes %d of %d; %.0f min\n', ...
    S.n_scored, n, S.n_contested, S.agreement, S.kappa, S.false_keep, S.n_ref_delete_scored, ...
    S.false_delete, S.n_ref_keep_scored, elapsed / 60);
results.report_html = acorn_training_report(neuron, ctx, key, results, o.out_dir, ...
    struct('headless', o.headless, 'open', ~o.headless));
end

function h = repo_hash(repo_root)
h = 'unknown';
try
    [stat, out] = system(sprintf('git -C "%s" rev-parse --short HEAD', repo_root));
    if stat == 0 && ~isempty(strtrim(out)); h = strtrim(out); end
catch
end
h = regexprep(h, '[^0-9a-f]', '');
if isempty(h); h = 'unknown'; end
end

function s = now_str()
s = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end

function p = strip_sep(p)
while numel(p) > 1 && (p(end) == '\' || p(end) == '/')
    p = p(1:end-1);
end
end

function v = getopt(o, name, default)
if isfield(o, name) && ~isempty(o.(name)); v = o.(name); else; v = default; end
end
