function acorn_training_selftest(scratch_dir, src_session_dir, key_path, video_bundle_dir)
% acorn_training_selftest -- headless tests for the training MATLAB code.
%
%   acorn_training_selftest(scratch_dir)                          pure tests
%   acorn_training_selftest(scratch_dir, src_session_dir, key_path)
%       also copies review_neuron.mat (+ Cn.mat) and the key into
%       <scratch_dir>/session/ and runs the static drill, report and progress
%       headless with scripted answers.  The source session folder is only
%       read.
%   acorn_training_selftest(..., video_bundle_dir)
%       also runs the video drill and the workflow drill headless on a folder
%       that already holds review_neuron.mat, training_key.mat and the raw
%       {session}.mat (e.g. a staged training bundle); results go to
%       <scratch_dir>/video_results, the bundle is only read.
%
% Run with:  matlab -batch "addpath(genpath('<repo>')); acorn_training_selftest('<scratch>')"
% (no '%' inside the -batch string).  Errors out when any check fails, so the
% exit code is non-zero.

fails = {};
    function check(cond, msg)
        if cond
            fprintf('  ok    %s\n', msg);
        else
            fprintf(2, '  FAIL  %s\n', msg);
            fails{end+1} = msg;
        end
    end

if ~exist(scratch_dir, 'dir'); mkdir(scratch_dir); end

%% ---- 1. scorer -------------------------------------------------------------
fprintf('[scorer]\n');
k = synth_key(16, [ones(8, 1); zeros(8, 1)]);
d = k.ref_keep;
S = acorn_training_score(d, zeros(16, 1), k);
check(S.agreement == 1 && S.kappa == 1 && S.n_scored == 16, 'identical decisions -> agreement 1, kappa 1');

S = acorn_training_score(ones(16, 1), zeros(16, 1), k);
check(abs(S.agreement - 0.5) < 1e-12 && abs(S.kappa) < 1e-12, 'all-keep vs half/half -> agreement 0.5, kappa 0');

% hand case: a=6 b=2 c=2 d=6 -> agreement 0.75, pe 0.5, kappa 0.5
d = [ones(6, 1); 0; 0; 1; 1; zeros(6, 1)];
S = acorn_training_score(d, zeros(16, 1), k);
check(S.a == 6 && S.b == 2 && S.c == 2 && S.d == 6, 'confusion counts a=6 b=2 c=2 d=6');
check(abs(S.agreement - 0.75) < 1e-12 && abs(S.pe - 0.5) < 1e-12 && abs(S.kappa - 0.5) < 1e-12, ...
    'agreement 0.75, pe 0.5, kappa 0.5');
check(isequal(S.false_keep_cols, [9; 10]) && isequal(S.false_delete_cols, [7; 8]), 'false keep/delete lists');
check(isequal(S.outcome([7 8 9 10 1 16]), [2; 2; 1; 1; 0; 0]), 'outcome codes');

k2 = k; k2.contested([1; 9]) = 1;
S = acorn_training_score(d, zeros(16, 1), k2);
check(S.n_scored == 14 && S.n_contested == 2 && S.false_keep == 1 && isequal(S.contested_cols, [1; 9]), ...
    'contested items excluded from scoring');
S = acorn_training_score(d, zeros(16, 1), k2, struct('exclude_contested', false));
check(S.n_scored == 16 && S.false_keep == 2, 'exclude_contested=false counts them');

dn = d; dn([3; 12]) = NaN;
S = acorn_training_score(dn, zeros(16, 1), k);
check(S.n_unvisited == 2 && isequal(S.unvisited_cols, [3; 12]) && S.decision_used(12) == 1 && S.b == 3, ...
    'NaN decisions = unvisited, counted as keep');

mask = false(16, 1); mask(1:8) = true;
S = acorn_training_score(d, zeros(16, 1), k, struct('subset_mask', mask));
check(S.n_subset == 8 && S.n_scored == 8 && S.c == 2 && S.b == 0 && all(S.outcome(9:16) == 4), 'subset mask');

km = k; km.ref_motion([11; 12]) = 1;
mot = zeros(16, 1); mot(11) = 1; mot(2) = 1;
dm = d; dm(2) = 0;
S = acorn_training_score(dm, mot, km);
check(S.motion_ref_n == 2 && S.motion_tagged_m == 1 && S.motion_deleted_any == 2 && S.motion_false_tags == 1, ...
    'motion stats (ref 2, tagged 1, deleted 2, false tag 1)');
km.has_motion_field = 0;
S = acorn_training_score(dm, mot, km);
check(isnan(S.motion_ref_n) && isnan(S.motion_false_tags), 'no motion field -> NaN motion stats');

S = acorn_training_score(d, zeros(16, 1), k, struct('subset_mask', false(16, 1)));
check(S.n_scored == 0 && isnan(S.agreement) && isnan(S.kappa), 'empty scored set -> NaN');

k3 = synth_key(4, [1; 1; 1; 1]);
S = acorn_training_score([1; 1; 1; 1], zeros(4, 1), k3);
check(S.agreement == 1 && isnan(S.kappa), 'no variance in either rater -> kappa NaN');

k4 = k; k4.hint_code(9) = 4; k4.hint_code(10) = 4; k4.hint_code(7) = 7;
S = acorn_training_score(d, zeros(16, 1), k4);
et = S.error_types;
check(et(et(:, 1) == 4, 2) == 2 && et(et(:, 1) == 7, 3) == 1, 'error types by hint code');

% contested side numbers: items 1 (ref keep) and 9 (ref delete) contested, model disagrees on both
k5 = k; k5.contested([1; 9]) = 1; k5.model_keep = 1 - k5.ref_keep;
d5 = d; d5(1) = 0; d5(9) = 1;                       % sides with the model on both
S = acorn_training_score(d5, zeros(16, 1), k5);
check(S.n_contested == 2 && S.contested_with_model == 2 && S.contested_with_ref == 0, 'contested: sided with the model on both');
d5(1) = 1;                                          % now with the reference on item 1
S = acorn_training_score(d5, zeros(16, 1), k5);
check(S.contested_with_ref == 1 && S.contested_with_model == 1 && S.n_scored == 14, 'contested: one each, still uncounted');

%% ---- 1b. transient finder (pure) ----------------------------------------------------------
fprintf('[transients]\n');
c = zeros(1, 300);
c(50:55) = [1.5 2 5 9 12 14];  c(56:100) = 14 * exp(-(1:45) / 8);      % onset 50 (1.5 > 10% of 14), peak 55
c(180:183) = [1 4 7 8];        c(184:230) = 8 * exp(-(1:47) / 8);       % onset 180, peak 183
[pk, on] = find_trace_transients(c, 5, 10);
check(isequal(pk, [55 183]) && isequal(on, [50 180]), 'peaks highest first, onsets at the first rising frame');
[pk, on] = find_trace_transients(c, 1, 10);
check(isequal(pk, 55) && isequal(on, 50), 'k limits the count');
[pk, on] = find_trace_transients(zeros(1, 50), 5, 10);
check(numel(pk) == 1 && on == pk, 'flat trace: one pseudo peak, onset at the peak');
[pk, on] = find_trace_transients(c(45:60), 5, 10);
check(pk == 11 && on == 6, 'works on a short window');

%% ---- 2. progress csv + paths ----------------------------------------------------------
fprintf('[progress csv]\n');
P = acorn_training_paths('Test Person!', fullfile(scratch_dir, 'acorn_training'));
check(strcmp(P.trainee, 'Test_Person') && exist(P.root, 'dir') == 7, 'paths: safe name, folder created');
csv = P.progress_csv;
if isfile(csv); delete(csv); end
row = struct('timestamp', '2026-10-01 12:00:00', 'trainee', 'Test_Person', 'area', 'BLA', 'task', 't', ...
    'session', 's1', 'stage', 1, 'mode', 'drill_coach', 'pass', '1', 'subset', 'all', 'n_items', 10, ...
    'n_visited', 10, 'n_scored', 9, 'n_contested', 1, 'n_ref_keep_scored', 4, 'n_ref_delete_scored', 5, ...
    'agreement', 0.8, 'kappa', 0.6, 'false_keep', 1, 'false_delete', 1, 'motion_ref_n', NaN, ...
    'motion_tagged_m', NaN, 'motion_deleted_any', NaN, 'motion_false_tags', NaN, 'n_flagged', 0, ...
    'median_sec', 3.5, 'total_sec', 40, 'result_file', 'drill_s1_x.mat', 'repo_hash', 'abc123');
acorn_training_append_progress(csv, row);
row.timestamp = '2026-10-01 13:00:00'; row.agreement = 0.9; row.result_file = 'drill_s1_y.mat';
acorn_training_append_progress(csv, row);
txt = fileread(csv);
lines = strsplit(strtrim(txt), '\n');
cols = acorn_training_progress_columns();
check(numel(lines) == 3 && strcmp(strtrim(lines{1}), strjoin(cols, ',')), 'header once, two rows');
check(~isempty(strfind(lines{2}, ',NaN,')) && ~isempty(strfind(lines{2}, '0.8000')), 'NaN and floats formatted');
% an older file (3 columns fewer) is extended in place, rows padded
old_csv = fullfile(P.root, 'progress_old.csv');
fid = fopen(old_csv, 'w');
fprintf(fid, '%s\n', strjoin(cols(1:end-3), ','));
fprintf(fid, '%s\n', strjoin(repmat({'1'}, 1, numel(cols) - 3), ','));
fclose(fid);
acorn_training_append_progress(old_csv, row);
ol = strsplit(strtrim(fileread(old_csv)), '\n');
check(numel(ol) == 3 && strcmp(strtrim(ol{1}), strjoin(cols, ',')) && sum(ol{2} == ',') == numel(cols) - 1 ...
      && sum(ol{3} == ',') == numel(cols) - 1, 'older progress.csv header extended, old row padded, new row appended');
fid = fopen(fullfile(P.root, 'progress_bad.csv'), 'w'); fprintf(fid, 'foo,bar\n1,2\n'); fclose(fid);
try
    acorn_training_append_progress(fullfile(P.root, 'progress_bad.csv'), row);
    check(false, 'unknown header refused');
catch err
    check(~isempty(strfind(err.message, 'unexpected header')), 'unknown header refused');
end
html = acorn_training_progress('Test Person!', struct('headless', true, 'root_override', fullfile(scratch_dir, 'acorn_training')));
check(isfile(html) && isfile(fullfile(fileparts(html), 'progress.png')), 'progress html + png written');

%% ---- 3. data tests (optional) -----------------------------------------------------------
if nargin >= 3 && ~isempty(src_session_dir) && ~isempty(key_path)
    fprintf('[drill on real data, headless]\n');
    sess = fullfile(scratch_dir, 'session');
    if ~exist(sess, 'dir'); mkdir(sess); end
    copyfile(fullfile(src_session_dir, 'review_neuron.mat'), fullfile(sess, 'review_neuron.mat'));
    copyfile(key_path, fullfile(sess, 'training_key.mat'));
    key = acorn_training_load_key(sess);
    check(key.n_review > 0 && numel(key.hint) == key.n_review && iscell(key.feature_names), 'key loads and validates');
    rn = load(fullfile(sess, 'review_neuron.mat'));
    ctx = acorn_training_context(rn.neuron, rn.Cn, key);
    check(numel(ctx.perm) == key.n_review && isequal(sort(ctx.perm), (1:key.n_review)'), 'display order is a permutation');
    [~, perm_ref] = sort(full(mean(rn.neuron.C, 2)), 'descend');
    check(isequal(ctx.perm, perm_ref(:)), 'perm equals the orderROIs(''mean'') sort');
    fig = figure('Visible', 'off', 'Position', [100 100 1280 560]);
    acorn_training_draw_panel(fig, rn.neuron, ctx, ctx.perm(1), 'Neuron 1', struct('show_cn', true, 'caption', 'test caption'));
    png = fullfile(scratch_dir, 'panel_test.png');
    print(fig, png, '-dpng', '-r60');
    close(fig);
    check(isfile(png), 'draw_panel renders headless with the Cn panel');
    n = key.n_review;
    out_dir = fullfile(sess, 'training_results');

    % exam mode, all keep: false keeps == number of scored reference deletes
    answers = repmat('k', 1, n);
    r1 = acorn_training_drill(sess, key, struct('trainee', 'Tester', 'mode', 'exam', 'subset', 'all', ...
        'drill_mode', 'static', 'show_cn', true, 'headless', true, 'answers', answers, 'out_dir', out_dir, 'stage', 2));
    check(strcmp(r1.drill_mode, 'static') && ~r1.video_pass, 'static drill records its mode');
    S1 = r1.score_final;
    check(S1.false_keep == S1.n_ref_delete_scored && S1.false_delete == 0, 'all-keep drill: false keeps = scored reference deletes');
    check(isfile(r1.result_file) && isfile(r1.report_html), 'result .mat and report html written');
    [~, rep_name] = fileparts(r1.report_html);
    files = dir(fullfile(out_dir, [rep_name, '_files'], '*.png'));
    check(numel(files) == min(60, S1.false_keep + S1.false_delete), 'one png per disagreement (capped at 60)');

    % coach mode, mirror the reference (perfect score), with an 'm' on a motion item
    answers = repmat('d', 1, n);
    for m = 1:n
        col = ctx.perm(m);
        if key.ref_keep(col) == 1; answers(m) = 'k';
        elseif key.ref_motion(col) == 1; answers(m) = 'm'; end
    end
    r2 = acorn_training_drill(sess, key, struct('trainee', 'Tester', 'mode', 'coach', 'subset', 'all', ...
        'drill_mode', 'static', 'show_cn', false, 'headless', true, 'answers', answers, 'out_dir', out_dir, 'stage', 2, ...
        'redo', [1 2 3]));
    S2 = r2.score_final;
    check(S2.agreement == 1 && S2.false_keep == 0 && S2.false_delete == 0, 'mirroring the reference scores perfectly');
    if key.has_motion_field == 1
        check(S2.motion_tagged_m == S2.motion_ref_n, 'all reference motion items tagged m');
    end
    check(strcmp(r2.score_basis, 'first') && r2.n_corrected == 3, 'coach mode scores the first decision; 3 redos counted');
    ch = r2.review_col(logical(r2.changed_after_feedback));
    check(numel(ch) == 3 && all(r2.decision_last_full(ch) ~= r2.decision_full(ch)) ...
          && all(r2.decision_last_full(setdiff(r2.review_col, ch)) == r2.decision_full(setdiff(r2.review_col, ch))), ...
          'last decisions differ from the scored ones only on the redone items');
    check(S2.contested_with_ref == S2.n_contested && S2.contested_with_model == 0, 'mirror: sided with the reference on every contested item');
    html2 = fileread(r2.report_html);
    check(~isempty(strfind(html2, 'changed after the feedback')) && ~isempty(strfind(html2, 'sided with the reference on')), ...
          'report shows the corrected count and the contested side numbers');
    % exam mode ignores the redo hook (no feedback, last decision scored)
    r2e = acorn_training_drill(sess, key, struct('trainee', 'Tester', 'mode', 'exam', 'subset', 'all', ...
        'drill_mode', 'static', 'show_cn', false, 'headless', true, 'answers', answers, 'out_dir', out_dir, 'stage', 2, ...
        'redo', [1 2 3]));
    check(strcmp(r2e.score_basis, 'last') && r2e.n_corrected == 0 && r2e.score_final.agreement == 1, 'exam mode: last decision scored, no corrections');

    % clear-cut subset
    nclear = sum(key.clear_cut);
    if nclear > 0
        r3 = acorn_training_drill(sess, key, struct('trainee', 'Tester', 'mode', 'exam', 'subset', 'clear', ...
            'drill_mode', 'static', 'show_cn', true, 'headless', true, 'answers', repmat('k', 1, nclear), 'out_dir', out_dir, 'stage', 1));
        check(r3.score_final.n_subset == nclear && r3.score_final.n_contested == 0, 'clear subset shows only clear-cut items');
    end
    rows = strsplit(strtrim(fileread(fullfile(out_dir, 'progress.csv'))), '\n');
    check(numel(rows) == 1 + 3 + (nclear > 0), 'session progress.csv has one row per attempt');
    check(~isempty(strfind(rows{2}, ',drill_static_exam,final,')), 'single-pass rows are tagged pass=final');
    check(~isempty(regexp(rows{3}, ',drill_static_coach,final,.*,3$', 'once')), 'coach row ends with n_corrected = 3');

    % a video drill without the video must refuse up front
    try
        acorn_training_drill(sess, key, struct('headless', true, 'answers', repmat('k', 1, n), 'out_dir', out_dir));
        check(false, 'video drill without a video refuses');
    catch err
        check(~isempty(strfind(err.message, 'raw video')), 'video drill without a video refuses');
    end
end

%% ---- 4. video + workflow drills (optional, needs a bundle with the raw video) ---------
if nargin >= 4 && ~isempty(video_bundle_dir)
    fprintf('[video drill on a staged bundle, headless]\n');
    keyv = acorn_training_load_key(video_bundle_dir);
    vout = fullfile(scratch_dir, 'video_results');
    rnv = load(fullfile(video_bundle_dir, 'review_neuron.mat'));
    ctxv = acorn_training_context(rnv.neuron, rnv.Cn, keyv);
    nv = keyv.n_review;
    mirror = repmat('d', 1, nv);
    for m = 1:nv
        col = ctxv.perm(m);
        if keyv.ref_keep(col) == 1; mirror(m) = 'k';
        elseif keyv.ref_motion(col) == 1; mirror(m) = 'm'; end
    end
    % video-only drill, mirroring the reference -> perfect
    rv = acorn_training_drill(video_bundle_dir, keyv, struct('trainee', 'Tester', 'mode', 'coach', ...
        'drill_mode', 'video', 'headless', true, 'answers', mirror, 'out_dir', vout, 'stage', 2));
    check(strcmp(rv.drill_mode, 'video') && rv.video_pass && isempty(rv.pass2), 'video drill: one pass, video used');
    check(rv.score_final.agreement == 1 && rv.score_final.false_delete == 0, 'video drill mirroring the reference scores perfectly');
    check(isfile(rv.report_html), 'video drill report written');
    % workflow drill: keep everything in triage (no real cells lost), then mirror in the video pass
    rw = acorn_training_drill(video_bundle_dir, keyv, struct('trainee', 'Tester', 'mode', 'exam', ...
        'drill_mode', 'workflow', 'headless', true, 'answers', repmat('k', 1, nv), 'answers2', mirror, ...
        'out_dir', vout, 'stage', 3));
    check(strcmp(rw.drill_mode, 'workflow') && ~isempty(rw.pass2) && numel(rw.pass2.review_col) == nv, ...
        'workflow drill: all triage keeps reach the video pass');
    check(rw.triage.false_delete == 0 && rw.score_final.agreement == 1, 'workflow drill: no real cells lost, final perfect');
    % workflow drill with one real cell deleted in triage: counted as lost, never reaches the video
    tri = repmat('k', 1, nv);
    kpos = find(keyv.ref_keep(ctxv.perm) == 1 & keyv.contested(ctxv.perm) == 0, 1);
    if ~isempty(kpos)
        tri(kpos) = 'd';
        rw2 = acorn_training_drill(video_bundle_dir, keyv, struct('trainee', 'Tester', 'mode', 'exam', ...
            'drill_mode', 'workflow', 'headless', true, 'answers', tri, 'answers2', repmat('k', 1, nv - 1), ...
            'out_dir', vout, 'stage', 3));
        check(rw2.triage.false_delete == 1 && numel(rw2.pass2.review_col) == nv - 1 && rw2.score_final.false_delete == 1, ...
            'workflow drill: a triage delete of a real cell is a lost cell and is excluded from the video pass');
    end
    rowsv = strsplit(strtrim(fileread(fullfile(vout, 'progress.csv'))), '\n');
    check(numel(rowsv) == 1 + 1 + 3 + 3 * (~isempty(kpos)), 'video/workflow progress rows: 1 + 3 per workflow attempt');
    check(~isempty(strfind(rowsv{3}, ',drill_workflow_exam,1,')) && ~isempty(strfind(rowsv{3}, ',NaN,NaN,NaN,')), ...
        'triage row carries NaN agreement/kappa/false_keep');
    clear rnv ctxv;
end

%% ---- summary ------------------------------------------------------------------------------
fprintf('\nRESULT: %d failure(s)\n', numel(fails));
for i = 1:numel(fails); fprintf(2, '  - %s\n', fails{i}); end
if ~isempty(fails)
    error('acorn_training_selftest: %d failure(s)', numel(fails));
end
end

function k = synth_key(n, ref_keep)
k = struct();
k.n_review = n;
k.review_col = (1:n)';
k.ref_keep = double(ref_keep(:));
k.model_keep = double(ref_keep(:));
k.ref_motion = zeros(n, 1);
k.has_motion_field = 1;
k.contested = zeros(n, 1);
k.clear_cut = ones(n, 1);
k.hint_code = zeros(n, 1);
k.model_score = 0.5 * ones(n, 1);
k.model_available = true;
k.hint = repmat({'-'}, n, 1);
end
