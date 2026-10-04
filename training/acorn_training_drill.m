function results = acorn_training_drill(session_dir, key, opts)
% acorn_training_drill -- re-review a session's candidates against the key.
%
%   results = acorn_training_drill(session_dir, key, opts)
%
% Records each keystroke per candidate WITHOUT touching the Sources2D object,
% scores the decisions against the key, writes
% training_results/drill_<session>_<ts>.mat, appends progress rows and renders
% an HTML report.
%
% opts.drill_mode (the important choice):
%   'video'    (default) every candidate is shown in the MOVIE panel -- the raw
%              video at the transient with the outline, the zoom and the trace
%              -- in one pass, and scored directly.  This is where the real
%              keep/delete judgement is made, so it is the training default.
%   'workflow' the real tool's order: a static triage pass (footprint + trace,
%              keep anything plausible), then the video pass over the survivors.
%              Pass 1 is reported only as "real cells lost before the video"
%              (false deletes); the headline numbers are the final pass.
%   'static'   footprint + trace only, no video.  Warm-up only: the reference
%              labels are post-video decisions, so a liberal static keep counts
%              against you here.  Used when the bundle has no video.
%
% Keys at the prompt: k or Enter = keep, d = delete, m = motion delete,
% b = back one, e = end the pass, x = flag "I disagree with the key",
% a number = jump to that shown neuron number.  In the movie panel also:
% n = next-biggest transient, t = back to the biggest -- both land pre_sec
% BEFORE the transient's onset so you can scrub or play through the rise;
% p = play from there through the peak to post_sec after it (twice); the
% slider scrubs.  No split / trim / delete-all.
%
% Other opts (all optional):
%   trainee     name (char)                        default 'trainee'
%   mode        'coach' (feedback after every decision) | 'exam' (at the end)
%   subset      'all' | 'clear' (key.clear_cut items only; falls back to all)
%   show_cn     correlation-image panel in the static view   default true
%   stage       stage number for the record                   default NaN
%   out_dir     default <session_dir>/training_results
%   progress_csv machine-level progress file (also appended)  default ''
%   exclude_contested  default true
%   pre_sec     seconds of lead-in before a transient's onset when you jump
%               to it (n / t) and when playback starts (p)     default 2
%   post_sec    seconds after the peak that playback runs to   default 3
%   headless    no visible figure, no web()                    default false
%   answers     char vector, one of 'kdm' per item in display order -- makes
%               the drill non-interactive (tests); answers2 = video pass of
%               the workflow mode; redo = item indices whose scripted first
%               decision is flipped after the feedback (tests, coach mode)
%   video_pass  legacy: true selects drill_mode 'workflow'
%
% What gets scored: in COACH mode the FIRST decision on each item, made
% before the feedback appeared; a change after feedback (press b, decide
% again) is the learning, counted separately as "corrected after feedback"
% and never improves the score.  In EXAM mode there is no feedback, so the
% last decision is scored.  Contested items (reference and model disagree)
% are never counted; the report shows how often you sided with each.

if nargin < 3 || isempty(opts); opts = struct(); end
o = struct();
o.trainee = getopt(opts, 'trainee', 'trainee');
o.mode = lower(getopt(opts, 'mode', 'coach'));
o.subset = lower(getopt(opts, 'subset', 'all'));
o.show_cn = logical(getopt(opts, 'show_cn', true));
o.stage = getopt(opts, 'stage', NaN);
o.out_dir = getopt(opts, 'out_dir', fullfile(session_dir, 'training_results'));
o.progress_csv = getopt(opts, 'progress_csv', '');
o.exclude_contested = logical(getopt(opts, 'exclude_contested', true));
o.headless = logical(getopt(opts, 'headless', false));
o.answers = char(getopt(opts, 'answers', ''));
o.answers2 = char(getopt(opts, 'answers2', ''));
o.redo = getopt(opts, 'redo', []);
o.pre_sec = double(getopt(opts, 'pre_sec', 2.0));      % ASSUMED default: 2 s before the onset
o.post_sec = double(getopt(opts, 'post_sec', 3.0));    % ASSUMED default: 3 s after the peak
dm = getopt(opts, 'drill_mode', '');
if isempty(dm)
    if logical(getopt(opts, 'video_pass', false)); dm = 'workflow'; else; dm = 'video'; end
end
o.drill_mode = lower(dm);
if ~any(strcmp(o.mode, {'coach', 'exam'}))
    error('acorn_training_drill: mode must be ''coach'' or ''exam''');
end
if ~any(strcmp(o.subset, {'all', 'clear'}))
    error('acorn_training_drill: subset must be ''all'' or ''clear''');
end
if ~any(strcmp(o.drill_mode, {'video', 'workflow', 'static'}))
    error('acorn_training_drill: drill_mode must be ''video'', ''workflow'' or ''static''');
end
session_dir = strip_sep(char(session_dir));
repo_root = fileparts(fileparts(mfilename('fullpath')));
need_video = ~strcmp(o.drill_mode, 'static');

% --- load the candidate set (read-only) -------------------------------------
rn = load(fullfile(session_dir, 'review_neuron.mat'));
neuron = rn.neuron;
Cn = [];
if isfield(rn, 'Cn'); Cn = rn.Cn; end
clear rn;
ctx = acorn_training_context(neuron, Cn, key);
n = ctx.n;
perm = ctx.perm;

% --- items: positions in display order --------------------------------------
if strcmp(o.subset, 'clear')
    items = find(key.clear_cut(perm) == 1);
    if isempty(items)
        fprintf(['No clear-cut items in this key (model scores unavailable?) -- ' ...
                 'drilling all %d candidates instead.\n'], n);
        items = (1:n)';
        o.subset = 'all';
    end
else
    items = (1:n)';
end
items = items(:);
n_items = numel(items);
scripted = ~isempty(o.answers);
if scripted && numel(o.answers) ~= n_items
    error('acorn_training_drill: opts.answers has %d entries, %d items shown', numel(o.answers), n_items);
end

% --- video (shared by the nested movie-panel functions) ------------------------
video = [];
num_frames = 0;
yflip = 0;
nam_mat = '';
if need_video
    nam_mat = locate_video(session_dir);
    if isempty(nam_mat)
        error(['acorn_training_drill: the raw video {session}.mat is not in %s. The ''%s'' drill needs it; ' ...
               'ask the operator for a bundle with the video, or use drill_mode ''static'' (warm-up only).'], ...
            session_dir, o.drill_mode);
    end
    fprintf('Loading the raw video (%s) ...\n', nam_mat);
    v = load(nam_mat, 'Y');
    video = flip(v.Y);                 % viewNeuronsVideo.m: flipped over the y axis
    clear v;
    yflip = size(video, 1);
    num_frames = size(video, 3);
    fprintf('  %d x %d x %d frames loaded.\n', size(video, 1), size(video, 2), num_frames);
end
scale = 2 / ctx.ssub;                  % viewNeuronsVideo.m contour scaling, mirrored
if strcmp(ctx.str_xlabel, 'Frame')
    pre_frames = 8; post_frames = 11;  % no frame rate known: treat the seconds as ~3.75 Hz frames
else
    dt = (ctx.t(end) - ctx.t(1)) / max(1, numel(ctx.t) - 1);
    pre_frames = max(2, round(o.pre_sec / dt));
    post_frames = max(2, round(o.post_sec / dt));
end
pass_cols = [];                        % review columns outlined in the movie panel
all_cx = []; all_cy = [];              % NaN-separated outline of pass_cols
state = struct('im', [], 'tline', [], 'slider', [], 'ax3', [], 'col', 0, ...
               'peaks', 1, 'onsets', 1, 'peak_idx', 1);

    function f = clip_start(j)
        f = max(1, state.onsets(j) - pre_frames);
    end

    function f = clip_end(j)
        f = min(num_frames, state.peaks(j) + post_frames);
    end

    function set_pass_cols(cols)
        pass_cols = cols(:)';
        all_cx = []; all_cy = [];
        for j = pass_cols
            c = ctx.contours{j};
            if isempty(c); continue; end
            all_cx = [all_cx, c(1, :) * scale, NaN]; %#ok<AGROW>
            all_cy = [all_cy, yflip - c(2, :) * scale, NaN]; %#ok<AGROW>
        end
    end

    function show_frame(f)
        f = max(1, min(num_frames, round(f)));
        if ~isempty(state.im) && isvalid(state.im); state.im.CData = video(:, :, f); end
        if ~isempty(state.tline) && isvalid(state.tline)
            state.tline.XData = [ctx.t(f), ctx.t(f)];
        end
        if ~isempty(state.slider) && isvalid(state.slider)
            state.slider.Value = (f - 1) / max(1, num_frames - 1);
        end
    end

    function scrub(src, ~)
        f = round(src.Value * (num_frames - 1)) + 1;
        show_frame(f);
    end

    function play_clip()
        % From pre_sec before the onset, through the peak, to post_sec after
        % it, twice; then rest on the peak so the cell stays visible.
        if o.headless || isempty(state.im) || ~isvalid(state.im); return; end
        j = state.peak_idx;
        frames = clip_start(j):clip_end(j);
        for rep = 1:2
            for f = frames
                if ~isvalid(state.im); return; end
                show_frame(f);
                drawnow;
                pause(0.12);
            end
        end
        show_frame(state.peaks(j));
    end

    function describe_transient(j)
        fprintf('  transient %d of %d: onset %.1f, peak %.1f (%s); showing %.1f before the onset -- p plays it, slider scrubs\n', ...
            j, numel(state.peaks), ctx.t(state.onsets(j)), ctx.t(state.peaks(j)), ...
            lower(ctx.str_xlabel), ctx.t(state.peaks(j)) - ctx.t(clip_start(j)));
    end

    function draw_movie(fig, col, label, tcol, caption)
        if nargin < 5; caption = ''; end
        if col ~= state.col
            [state.peaks, state.onsets] = find_trace_transients(full(neuron.C(col, :)), 5, 10);
            state.peak_idx = 1;
            state.col = col;
        end
        pk = state.peaks(state.peak_idx);
        set(0, 'CurrentFigure', fig);
        clf(fig);
        ax1 = subplot(2, 2, 1, 'Parent', fig);
        state.im = imagesc(video(:, :, pk));
        axis equal; axis off; hold(ax1, 'on');
        if ~isempty(all_cx)
            plot(ax1, all_cx, all_cy, '-k');
        end
        c = ctx.contours{col};
        if ~isempty(c)
            plot(ax1, c(1, :) * scale, yflip - c(2, :) * scale, '-r', 'LineWidth', 1.5);
        end
        hold(ax1, 'off');
        title(sprintf('%s   transient %d of %d (peak at %.1f %s)', label, state.peak_idx, ...
            numel(state.peaks), ctx.t(pk), lower(ctx.str_xlabel)), 'color', tcol, 'Interpreter', 'none');
        subplot(2, 2, 2, 'Parent', fig);
        neuron.image(neuron.A(:, col) .* ctx.Amask(:, col));
        axis equal; axis off;
        xlim(ctx.ctr(col, 2) + [-ctx.gSiz, ctx.gSiz] * 2);
        ylim(ctx.ctr(col, 1) + [-ctx.gSiz, ctx.gSiz] * 2);
        title('zoom');
        ax3 = subplot(2, 2, 3:4, 'Parent', fig);
        plot(ax3, ctx.t, full(neuron.C_raw(col, :)) * ctx.Amax(col), 'linewidth', 2); hold(ax3, 'on');
        plot(ax3, ctx.t, full(neuron.C(col, :)) * ctx.Amax(col), 'r');
        xlim(ax3, [ctx.t(1), ctx.t(end)]); xlabel(ax3, ctx.str_xlabel);
        yl = get(ax3, 'ylim');
        state.tline = plot(ax3, [ctx.t(pk), ctx.t(pk)], yl, 'y', 'LineWidth', 1.5);
        hold(ax3, 'off');
        state.ax3 = ax3;
        state.slider = [];
        if ~o.headless
            pos3 = ax3.Position;
            state.slider = uicontrol('Parent', fig, 'Style', 'slider', 'Units', 'normalized', ...
                'Position', [pos3(1), pos3(2) + 0.37, pos3(3), 0.04], ...
                'Min', 0, 'Max', 1, 'Value', (pk - 1) / max(1, num_frames - 1), ...
                'SliderStep', [1 / max(1, num_frames), 0.01], 'Callback', @scrub);
        end
        if ~isempty(caption)
            annotation(fig, 'textbox', [0.005, 0.935, 0.99, 0.06], 'String', caption, ...
                'Interpreter', 'none', 'FontSize', 9, 'EdgeColor', 'none', ...
                'BackgroundColor', [1, 1, 0.85], 'FitBoxToText', 'off');
        end
    end

    function consumed = movie_key(k)
        % n / t land BEFORE the onset (pre_sec of lead-in) so the rise can be
        % scrubbed or played; p plays from there through the decay.
        consumed = true;
        switch k
            case 'p'
                play_clip();
            case 'n'
                state.peak_idx = mod(state.peak_idx, numel(state.peaks)) + 1;
                show_frame(clip_start(state.peak_idx));
                describe_transient(state.peak_idx);
            case 't'
                state.peak_idx = 1;
                show_frame(clip_start(1));
                describe_transient(1);
            otherwise
                consumed = false;
        end
    end

movie = struct('draw', @draw_movie, 'onkey', @movie_key, 'set_cols', @set_pass_cols);

if o.headless; vis = 'off'; else; vis = 'on'; end
started_at = now_str();
t_start = tic;
if ~o.headless
    fprintf('\n=== ACORN training drill (%s): %s / %s / %s ===\n', o.drill_mode, key.area, key.task, key.session);
    fprintf('%d candidate(s) in this pass (%s), feedback mode %s. Reference reviewer: %s.\n', ...
        n_items, o.subset, o.mode, key.reference_reviewer);
    fprintf('Keys: k/Enter keep, d delete, m motion delete, b back, e end, x flag, number = jump');
    if need_video
        fprintf(', n next transient / t biggest (lands %.0f s before the onset), p play through it', o.pre_sec);
    end
    fprintf('.\n\n');
end

subset_mask = false(n, 1);
subset_mask(perm(items)) = true;
sopts = struct('subset_mask', subset_mask, 'exclude_contested', o.exclude_contested);
score_first = strcmp(o.mode, 'coach');     % coach: the decision BEFORE the feedback is scored
results = struct();
results.drill_mode = o.drill_mode;
if score_first; results.score_basis = 'first'; else; results.score_basis = 'last'; end
results.pass2 = [];
results.score2 = [];
results.triage = [];

switch o.drill_mode
    case 'static'
        fig = figure('Visible', vis, 'Position', [100, 100, 1280, 560], 'Name', 'ACORN training drill (static)', 'NumberTitle', 'off');
        pc = struct('panel', 'static', 'numbering', 'pos', 'feedback', 'full');
        st = pass_loop(fig, neuron, ctx, key, items, perm, o, scripted, o.answers, pc, []);
        close_if_valid(fig);
        [decision, motion] = to_full(st, items, perm, n, score_first);
        [dlast, mlast] = to_full(st, items, perm, n, false);
        S = acorn_training_score(decision, motion, key, sopts);
        fill_pass(st, items, perm);
        results.score = S;
        results.score_final = S;
        results.decision_full = decision; results.motion_full = motion;
        results.decision_last_full = dlast; results.motion_last_full = mlast;

    case 'video'
        fig = figure('Visible', vis, 'Position', [100, 100, 1024, 512], 'Name', 'ACORN training drill (video)', 'NumberTitle', 'off');
        set_pass_cols(perm(items));
        pc = struct('panel', 'movie', 'numbering', 'pos', 'feedback', 'full');
        st = pass_loop(fig, neuron, ctx, key, items, perm, o, scripted, o.answers, pc, movie);
        close_if_valid(fig);
        [decision, motion] = to_full(st, items, perm, n, score_first);
        [dlast, mlast] = to_full(st, items, perm, n, false);
        S = acorn_training_score(decision, motion, key, sopts);
        fill_pass(st, items, perm);
        results.score = S;
        results.score_final = S;
        results.decision_full = decision; results.motion_full = motion;
        results.decision_last_full = dlast; results.motion_last_full = mlast;

    case 'workflow'
        fig = figure('Visible', vis, 'Position', [100, 100, 1280, 560], 'Name', 'ACORN training drill (static triage)', 'NumberTitle', 'off');
        pc = struct('panel', 'static', 'numbering', 'pos', 'feedback', 'triage');
        if ~o.headless
            fprintf('--- Pass 1, static triage: keep anything plausible; the video pass decides. ---\n');
        end
        st = pass_loop(fig, neuron, ctx, key, items, perm, o, scripted, o.answers, pc, []);
        close_if_valid(fig);
        [decision, motion] = to_full(st, items, perm, n, score_first);
        [dlast, mlast] = to_full(st, items, perm, n, false);
        S1 = acorn_training_score(decision, motion, key, sopts);
        fill_pass(st, items, perm);
        results.triage = S1;
        keep_items = items(st.dec ~= 0);           % what you ended up keeping survives (last decisions)
        decision_final = decision; motion_final = motion;
        dlast_final = dlast; mlast_final = mlast;
        if isempty(keep_items)
            fprintf(2, 'Nothing kept in pass 1 -- no video pass.\n');
        else
            scripted2 = ~isempty(o.answers2);
            if scripted2 && numel(o.answers2) ~= numel(keep_items)
                error('acorn_training_drill: opts.answers2 has %d entries, %d items in the video pass', ...
                    numel(o.answers2), numel(keep_items));
            end
            if ~o.headless
                fprintf('\n--- Pass 2, video: %d kept candidate(s). Watch the transient; press m when the frame moves instead of a cell firing. ---\n', numel(keep_items));
            end
            fig2 = figure('Visible', vis, 'Position', [100, 100, 1024, 512], 'Name', 'ACORN training drill (video pass)', 'NumberTitle', 'off');
            set_pass_cols(perm(keep_items));
            pc2 = struct('panel', 'movie', 'numbering', 'seq', 'feedback', 'full');
            st2 = pass_loop(fig2, neuron, ctx, key, keep_items, perm, o, scripted2, o.answers2, pc2, movie);
            close_if_valid(fig2);
            cols2 = perm(keep_items);
            for m = 1:numel(keep_items)
                if st2.visited(m)
                    if score_first
                        decision_final(cols2(m)) = st2.first_dec(m);
                        motion_final(cols2(m)) = st2.first_mot(m);
                    else
                        decision_final(cols2(m)) = st2.dec(m);
                        motion_final(cols2(m)) = st2.mot(m);
                    end
                    dlast_final(cols2(m)) = st2.dec(m);
                    mlast_final(cols2(m)) = st2.mot(m);
                end
            end
            mask2 = false(n, 1); mask2(cols2) = true;
            p2 = struct();
            p2.review_col = cols2(:);
            p2.shown_pos2 = (1:numel(keep_items))';
            p2.pass1_pos = keep_items(:);
            p2.decision2 = int8(st2.dec);
            p2.motion2 = int8(st2.mot);
            p2.first_decision2 = st2.first_dec;
            p2.first_motion2 = st2.first_mot;
            p2.changed_after_feedback2 = st2.changed;
            p2.n_corrected2 = sum(st2.changed);
            p2.visited2 = st2.visited;
            p2.seconds2 = st2.secs;
            p2.flagged2 = st2.flagged;
            p2.video_file = nam_mat;
            results.pass2 = p2;
            results.score2 = acorn_training_score(decision_final, motion_final, key, ...
                struct('subset_mask', mask2, 'exclude_contested', o.exclude_contested));
        end
        results.score = S1;
        results.score_final = acorn_training_score(decision_final, motion_final, key, sopts);
        results.decision_full = decision_final; results.motion_full = motion_final;
        results.decision_last_full = dlast_final; results.motion_last_full = mlast_final;
end
clear video;

ts = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
results.schema_version = 2;
results.trainee = o.trainee;
results.area = key.area; results.task = key.task; results.session = key.session;
results.stage = o.stage;
results.mode = o.mode;
results.subset = o.subset;
results.show_cn = o.show_cn;
results.video_pass = need_video;
results.video_file = nam_mat;
results.started_at = started_at;
results.ts = ts;
results.repo_hash = repo_hash(repo_root);
results.matlab_version = version;
results.n_review = n;
results.perm = perm;
results.subset_mask = subset_mask;
results.key_path = key.path;
results.finished_at = now_str();
results.total_sec = toc(t_start);

% --- save ----------------------------------------------------------------------------
if ~exist(o.out_dir, 'dir'); mkdir(o.out_dir); end
result_file = fullfile(o.out_dir, sprintf('drill_%s_%s.mat', key.session, ts));
opts_saved = o;
opts_saved.answers = '';
opts_saved.answers2 = '';
save(result_file, 'results', 'key', 'opts_saved', '-v7');
results.result_file = result_file;

% --- progress rows ---------------------------------------------------------------------
mode_tag = sprintf('drill_%s_%s', o.drill_mode, o.mode);
n_corr1 = sum(results.changed_after_feedback);
n_corr2 = 0;
if ~isempty(results.pass2); n_corr2 = results.pass2.n_corrected2; end
results.n_corrected = n_corr1 + n_corr2;
rows = {};
if strcmp(o.drill_mode, 'workflow')
    rows{end+1} = progress_row(results, results.triage, '1', results.seconds, results.flagged, mode_tag, true, n_corr1);
    if ~isempty(results.pass2)
        rows{end+1} = progress_row(results, results.score2, '2', results.pass2.seconds2, results.pass2.flagged2, mode_tag, false, n_corr2);
    end
    rows{end+1} = progress_row(results, results.score_final, 'final', all_seconds(results), results.flagged | any_flag(results), mode_tag, false, n_corr1 + n_corr2);
else
    rows{end+1} = progress_row(results, results.score_final, 'final', results.seconds, results.flagged, mode_tag, false, n_corr1);
end
for i = 1:numel(rows)
    acorn_training_append_progress(fullfile(o.out_dir, 'progress.csv'), rows{i});
    if ~isempty(o.progress_csv)
        acorn_training_append_progress(o.progress_csv, rows{i});
    end
end

% --- console summary + report -----------------------------------------------------------
Sf = results.score_final;
fprintf('\n--- Result (%s, %s drill) ---\n', key.session, o.drill_mode);
if strcmp(o.drill_mode, 'workflow')
    fprintf('static triage: %d real cell(s) deleted before the video pass (of %d reference keeps shown)\n', ...
        results.triage.false_delete, results.triage.n_ref_keep_scored);
end
fprintf('scored %d of %d shown (%d contested not counted, %d never visited)\n', ...
    Sf.n_scored, Sf.n_subset, Sf.n_contested, Sf.n_unvisited);
fprintf('agreement %.3f   kappa %.3f   false keeps %d of %d reference deletes   false deletes %d of %d reference keeps\n', ...
    Sf.agreement, Sf.kappa, Sf.false_keep, Sf.n_ref_delete_scored, Sf.false_delete, Sf.n_ref_keep_scored);
if key.has_motion_field == 1
    fprintf('motion: %d reference motion deletes; you tagged %d with m, deleted %d by any key; %d m-tags on non-motion items\n', ...
        Sf.motion_ref_n, Sf.motion_tagged_m, Sf.motion_deleted_any, Sf.motion_false_tags);
end
if Sf.n_contested > 0
    fprintf('contested items (not counted): you sided with the reference on %d and with the model on %d of %d\n', ...
        Sf.contested_with_ref, Sf.contested_with_model, Sf.n_contested);
end
if score_first
    fprintf('scored on your FIRST decisions; %d decision(s) changed after feedback (not scored)\n', results.n_corrected);
end
fprintf('saved %s\n', result_file);
results.report_html = acorn_training_report(neuron, ctx, key, results, o.out_dir, ...
    struct('headless', o.headless, 'open', ~o.headless));

% ---- nested helpers that need the results struct ------------------------------------
    function fill_pass(st, items_, perm_)
        results.review_col = perm_(items_);
        results.shown_pos = items_;
        results.decision = int8(st.dec);            % last decision per item
        results.motion = int8(st.mot);
        results.first_decision = st.first_dec;      % first decision (NaN = unvisited)
        results.first_motion = st.first_mot;
        results.visited = st.visited;
        results.n_views = st.n_views;
        results.seconds = st.secs;
        results.flagged = st.flagged;
        results.changed_after_feedback = st.changed;
    end
end

% =====================================================================================
function st = pass_loop(fig, neuron, ctx, key, items, perm, o, scripted, answers, pc, movie)
% One pass over items (positions).  pc.panel 'static'|'movie', pc.numbering
% 'pos' (the Neuron N of the full set) | 'seq' (1..n of this pass),
% pc.feedback 'full' | 'triage'.  Returns per-item state.
n_items = numel(items);
st = struct();
st.dec = -ones(n_items, 1);
st.mot = zeros(n_items, 1);
st.visited = false(n_items, 1);
st.n_views = zeros(n_items, 1);
st.secs = zeros(n_items, 1);
st.flagged = false(n_items, 1);
st.changed = false(n_items, 1);
st.first_dec = nan(n_items, 1);      % decision before any feedback (coach mode scores this)
st.first_mot = zeros(n_items, 1);
redo = [];
if isfield(o, 'redo') && ~isempty(o.redo); redo = double(o.redo(:))'; end
coach = strcmp(o.mode, 'coach');
use_movie = strcmp(pc.panel, 'movie');
m = 1;
while m >= 1 && m <= n_items
    pos = items(m);
    col = perm(pos);
    if strcmp(pc.numbering, 'pos')
        shown = pos;
        label = sprintf('Neuron %d   [item %d of %d]', pos, m, n_items);
    else
        shown = m;
        label = sprintf('Neuron %d (video pass)   [item %d of %d]', m, m, n_items);
    end
    tcol = 'k';
    if st.dec(m) == 0; tcol = 'r'; end
    if use_movie
        movie.draw(fig, col, label, tcol);
    else
        acorn_training_draw_panel(fig, neuron, ctx, col, label, struct('show_cn', o.show_cn, 'title_color', tcol));
    end
    drawnow;
    st.n_views(m) = st.n_views(m) + 1;
    t0 = tic;
    if scripted
        temp = answers(m);
    else
        if use_movie
            fprintf('Neuron %d, keep(k, default)/delete(d)/MOTION delete(m)/play(p)/next transient(n)/backward(b)/end(e)/flag(x)/jump to(#):    ', shown);
        else
            fprintf('Neuron %d, keep(k, default)/delete(d)/MOTION delete(m)/backward(b)/end(e)/flag(x)/jump to(#):    ', shown);
        end
        temp = strtrim(input('', 's'));
    end
    st.secs(m) = st.secs(m) + toc(t0);
    decided = false;
    switch lower(temp)
        case {'', 'k'}
            st.dec(m) = 1; st.mot(m) = 0; decided = true;
        case 'd'
            st.dec(m) = 0; st.mot(m) = 0; decided = true;
        case 'm'
            st.dec(m) = 0; st.mot(m) = 1; decided = true;
        case 'b'
            m = max(1, m - 1);
            continue;
        case 'e'
            break;
        case 'x'
            st.flagged(m) = ~st.flagged(m);
            if st.flagged(m); fprintf('  flagged: you disagree with the key on this one.\n');
            else; fprintf('  flag removed.\n'); end
            continue;
        case {'p', 'n', 't'}
            if use_movie
                movie.onkey(lower(temp));
            else
                fprintf('  (%s is a movie-panel key; not available in the static view)\n', lower(temp));
            end
            continue;
        otherwise
            jump = str2double(temp);
            if ~isnan(jump)
                if strcmp(pc.numbering, 'pos')
                    idx = find(items == jump, 1);
                else
                    idx = jump; if idx < 1 || idx > n_items; idx = []; end
                end
                if isempty(idx)
                    fprintf('There is no neuron %s in this pass -- staying put.\n', temp);
                else
                    m = idx;
                end
                continue;
            end
            st.dec(m) = 1; st.mot(m) = 0; decided = true;   % viewNeurons: unknown input = keep
    end
    if decided
        st.visited(m) = true;
        if isnan(st.first_dec(m))
            st.first_dec(m) = st.dec(m);
            st.first_mot(m) = st.mot(m);
        end
        st.changed(m) = (st.first_dec(m) ~= st.dec(m)) || (st.first_mot(m) ~= st.mot(m));
        if coach
            fb = feedback_text(key, col, st.dec(m), st.mot(m), pc.feedback);
            fprintf('  >> %s\n', fb);
            if scripted && any(redo == m)
                % test hook: pretend the trainee pressed b and flipped the decision
                st.dec(m) = 1 - st.first_dec(m); st.mot(m) = 0;
                st.changed(m) = true;
            end
            if ~scripted
                if use_movie
                    movie.draw(fig, col, label, tcol, fb);
                else
                    acorn_training_draw_panel(fig, neuron, ctx, col, label, ...
                        struct('show_cn', o.show_cn, 'title_color', tcol, 'caption', fb));
                end
                drawnow;
                resp = strtrim(input('  [Enter] next / b = redo this one / x = flag "I disagree":  ', 's'));
                if strcmpi(resp, 'b')
                    continue;
                elseif strcmpi(resp, 'x')
                    st.flagged(m) = true;
                    fprintf('  flagged.\n');
                end
            end
        end
        m = m + 1;
    end
end
end

function s = feedback_text(key, col, d, mt, kind)
% kind 'full': reveal the reference, model and hint.  kind 'triage' (static
% pass of the workflow drill): a keep is neutral -- the video decides -- and
% only a delete is judged, because a real cell deleted here never reaches
% the video pass.
if d == 1; you = 'keep'; elseif mt == 1; you = 'delete (m)'; else; you = 'delete'; end
if strcmp(kind, 'triage') && d == 1
    s = 'You: keep -- fine for the static pass; the video pass decides.';
    return;
end
if key.ref_keep(col) == 1; ref = 'keep'; else; ref = 'delete'; end
if key.ref_motion(col) == 1; ref = [ref, ' (motion)']; end
if key.model_available && ~isnan(key.model_score(col))
    model = sprintf('%.2f', key.model_score(col));
else
    model = 'n/a';
end
if key.contested(col) == 1
    verdict = 'CONTESTED (not counted)';
elseif d == key.ref_keep(col)
    verdict = 'agree';
elseif strcmp(kind, 'triage')
    verdict = 'REAL CELL LOST: the reference kept this; in the static pass keep anything plausible';
else
    verdict = 'DISAGREE';
end
s = sprintf('You: %s | Reference: %s | Model: %s | %s | %s', you, ref, model, verdict, key.hint{col});
end

function [decision, motion] = to_full(st, items, perm, n, use_first)
% Per-review-column vectors (NaN = unvisited).  use_first: the decision made
% before any feedback (coach mode scoring); otherwise the last decision.
if nargin < 5; use_first = false; end
decision = nan(n, 1);
motion = zeros(n, 1);
for m = 1:numel(items)
    col = perm(items(m));
    if st.visited(m)
        if use_first
            decision(col) = st.first_dec(m);
            motion(col) = st.first_mot(m);
        else
            decision(col) = st.dec(m);
            motion(col) = st.mot(m);
        end
    end
end
end

function nam = locate_video(session_dir)
% Same lookup as CNMFe_final_save.m: <session>.mat, else any .mat that is not
% one of the pipeline's outputs.
[~, session_nm] = fileparts(session_dir);
nam = fullfile(session_dir, [session_nm, '.mat']);
if isfile(nam); return; end
files = dir(fullfile(session_dir, '*.mat'));
exclude = {'neuron', 'Cn', 'Coor', 'pnr', 'Ybg', 'Ybg_mean', 'Ybg_weights', ...
           'spatial_footprints', 'review_neuron', 'training_key', 'labels', ...
           'review_checkpoint', 'motion_qc', 'motion_series', 'motion_vec', 'cand_traces'};
nam = '';
for i = 1:numel(files)
    [~, nm] = fileparts(files(i).name);
    if any(strcmp(nm, exclude)) || startsWith(nm, 'merge_log') || startsWith(nm, 'drill_') ...
            || startsWith(nm, 'rehearsal_')
        continue;
    end
    nam = fullfile(session_dir, files(i).name);
    return;
end
end

% =====================================================================================
function row = progress_row(results, S, pass_tag, secs, flagged, mode_tag, triage, n_corrected)
% triage = true: the static pass of the workflow drill.  Only "real cells lost
% before the video" (false deletes) is a meaningful number there; agreement,
% kappa, false keeps and motion counts are written NaN so nobody reads them
% as a result.  n_corrected = decisions changed after feedback in this pass.
if nargin < 8; n_corrected = 0; end
row = struct();
row.timestamp = results.finished_at;
row.trainee = results.trainee;
row.area = results.area; row.task = results.task; row.session = results.session;
row.stage = results.stage;
row.mode = mode_tag;
row.pass = pass_tag;
row.subset = results.subset;
row.n_items = S.n_subset;
row.n_visited = S.n_subset - S.n_unvisited;
row.n_scored = S.n_scored;
row.n_contested = S.n_contested;
row.n_ref_keep_scored = S.n_ref_keep_scored;
row.n_ref_delete_scored = S.n_ref_delete_scored;
if triage
    row.agreement = NaN; row.kappa = NaN; row.false_keep = NaN;
    row.motion_ref_n = NaN; row.motion_tagged_m = NaN;
    row.motion_deleted_any = NaN; row.motion_false_tags = NaN;
else
    row.agreement = S.agreement;
    row.kappa = S.kappa;
    row.false_keep = S.false_keep;
    row.motion_ref_n = S.motion_ref_n;
    row.motion_tagged_m = S.motion_tagged_m;
    row.motion_deleted_any = S.motion_deleted_any;
    row.motion_false_tags = S.motion_false_tags;
end
row.false_delete = S.false_delete;
row.n_flagged = sum(flagged);
sv = secs(secs > 0);
if isempty(sv); row.median_sec = NaN; else; row.median_sec = median(sv); end
row.total_sec = sum(secs);
[~, nm, ext] = fileparts(results.result_file);
row.result_file = [nm, ext];
row.repo_hash = results.repo_hash;
row.contested_with_ref = S.contested_with_ref;
row.contested_with_model = S.contested_with_model;
row.n_corrected = n_corrected;
end

function s = all_seconds(results)
s = results.seconds;
if ~isempty(results.pass2); s = [s; results.pass2.seconds2]; end
end

function f = any_flag(results)
f = false(size(results.flagged));
if ~isempty(results.pass2)
    idx = results.pass2.pass1_pos;
    for i = 1:numel(idx)
        k = find(results.shown_pos == idx(i), 1);
        if ~isempty(k) && results.pass2.flagged2(i); f(k) = true; end
    end
end
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

function close_if_valid(fig)
if ishandle(fig) && isvalid(fig); close(fig); end
end

function v = getopt(o, name, default)
if isfield(o, name) && ~isempty(o.(name))
    v = o.(name);
else
    v = default;
end
end
