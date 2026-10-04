function html = acorn_training_report(neuron, ctx, key, results, out_dir, opts)
% acorn_training_report -- HTML report for one drill or rehearsal attempt.
%
%   html = acorn_training_report(neuron, ctx, key, results, out_dir, opts)
%
% Writes <out_dir>/report_<session>_<ts>.html plus one PNG per disagreement
% in <out_dir>/report_<session>_<ts>_files/.  Each disagreement panel is the
% same drawing the drill showed, captioned with your decision, the reference
% decision, the model score and the hint.  Contested items (reference and
% model disagree) are listed separately and never counted.  Static HTML, no
% scripts; opens in the system browser unless opts.headless / ~opts.open.
% opts.max_panels (default 60) caps the number of images.

if nargin < 6 || isempty(opts); opts = struct(); end
headless = logical(getopt(opts, 'headless', false));
do_open = logical(getopt(opts, 'open', ~headless));
max_panels = getopt(opts, 'max_panels', 60);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

name = sprintf('report_%s_%s', results.session, results.ts);
img_dir = fullfile(out_dir, [name, '_files']);
if ~exist(img_dir, 'dir'); mkdir(img_dir); end
html = fullfile(out_dir, [name, '.html']);

S = results.score_final;
n = ctx.n;
dec = results.decision_full;      % n x 1, NaN = unvisited (the SCORED decisions)
mot = results.motion_full;
if isfield(results, 'decision_last_full') && ~isempty(results.decision_last_full)
    dlast = results.decision_last_full; mlast = results.motion_last_full;
else
    dlast = dec; mlast = mot;
end
score_basis = 'last';
if isfield(results, 'score_basis') && ~isempty(results.score_basis); score_basis = results.score_basis; end
n_corrected = 0;
if isfield(results, 'n_corrected') && ~isempty(results.n_corrected); n_corrected = results.n_corrected; end
changed_cols = [];
if isfield(results, 'changed_after_feedback')
    changed_cols = results.review_col(logical(results.changed_after_feedback));
end
flag_cols = results.review_col(results.flagged);
if isfield(results, 'pass2') && ~isempty(results.pass2)
    flag_cols = unique([flag_cols(:); results.pass2.review_col(results.pass2.flagged2)]);
    if isfield(results.pass2, 'changed_after_feedback2')
        changed_cols = unique([changed_cols(:); results.pass2.review_col(logical(results.pass2.changed_after_feedback2))]);
    end
end

fid = fopen(html, 'w');
if fid < 0; error('acorn_training_report: cannot write %s', html); end
cleaner = onCleanup(@() fclose(fid));
w = @(varargin) fprintf(fid, varargin{:});

w('<!DOCTYPE html>\n<html><head><meta charset="ascii"><title>ACORN training report %s</title>\n', esc(results.session));
w('<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;max-width:1300px;color:#222}');
w('table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:4px 8px;text-align:left;font-size:13px}');
w('th{background:#eee}.panel{border:1px solid #ccc;margin:14px 0;padding:8px}.panel img{max-width:100%%}');
w('.fk{border-left:6px solid #d9534f}.fd{border-left:6px solid #f0ad4e}.con{border-left:6px solid #5bc0de}');
w('.small{color:#666;font-size:12px}.hint{background:#fffbe6;padding:4px 6px;display:inline-block}</style></head><body>\n');

drill_mode = 'static';
if isfield(results, 'drill_mode') && ~isempty(results.drill_mode); drill_mode = results.drill_mode; end
is_workflow = strcmp(drill_mode, 'workflow') && isfield(results, 'triage') && ~isempty(results.triage);
w('<h1>ACORN training report</h1>\n');
w('<p><b>%s</b> / %s / %s<br>', esc(results.area), esc(results.task), esc(results.session));
w('Trainee: <b>%s</b> &nbsp; Drill: %s &nbsp; Feedback: %s &nbsp; Subset: %s &nbsp; Stage: %s &nbsp; Finished: %s<br>', ...
    esc(results.trainee), esc(drill_mode), esc(results.mode), esc(results.subset), num_or_dash(results.stage), esc(results.finished_at));
if strcmp(score_basis, 'first')
    w('Scored on your <b>first</b> decision on each item (before the feedback); decisions changed after feedback are listed, not scored.<br>');
end
w('Reference reviewer: <b>%s</b> &nbsp; Model: %s (%s, %s, trained on %s sessions)</p>\n', ...
    esc(key.reference_reviewer), esc(key.model_type), esc(key.score_kind), ...
    esc(key.area), num_or_dash(key.model_n_sessions));

% --- summary -------------------------------------------------------------------------
w('<h2>Summary</h2>\n');
if is_workflow
    w(['<p>Workflow drill: pass 1 is the static triage (keep anything plausible). Only one number matters there: ' ...
       '<b>real cells lost before the video</b> (false deletes). The final column is what counts.</p>\n']);
elseif strcmp(drill_mode, 'static')
    w(['<p class="small">Static drill: the reference labels are post-video decisions, so a liberal keep that the video ' ...
       'would have removed counts as a false keep here. Use the video drill for the real judgement.</p>\n']);
end
w('<table><tr><th></th>');
cols = {};
triage_col = false(1, 0);
if is_workflow
    cols = {{'pass 1 (static triage)', results.triage}, {'final (after the video pass)', results.score_final}};
    triage_col = [true, false];
    if isfield(results, 'score2') && ~isempty(results.score2)
        cols = {cols{1}, {'pass 2 (video, your keeps only)', results.score2}, cols{2}};
        triage_col = [true, false, false];
    end
else
    cols = {{'this attempt', results.score_final}};
    triage_col = false;
end
for i = 1:numel(cols); w('<th>%s</th>', esc(cols{i}{1})); end
w('</tr>\n');
triage_rows = {'n_subset', 'n_unvisited', 'n_contested', 'n_scored', 'false_delete', 'n_ref_keep_scored', ...
               'contested_with_ref', 'contested_with_model'};
rows = {'items shown', 'n_subset'; 'never visited (counted as keep)', 'n_unvisited'; ...
        'contested (not counted)', 'n_contested'; 'scored', 'n_scored'; ...
        'agreement', 'agreement'; 'Cohen''s kappa', 'kappa'; ...
        'false keeps (you kept, reference deleted)', 'false_keep'; ...
        'reference deletes among scored', 'n_ref_delete_scored'; ...
        'false deletes (you deleted, reference kept)', 'false_delete'; ...
        'reference keeps among scored', 'n_ref_keep_scored'; ...
        'reference motion deletes', 'motion_ref_n'; 'of which you tagged m', 'motion_tagged_m'; ...
        'of which you deleted (any key)', 'motion_deleted_any'; 'm-tags on non-motion items', 'motion_false_tags'; ...
        'contested items: you sided with the reference (not counted)', 'contested_with_ref'; ...
        'contested items: you sided with the model (not counted)', 'contested_with_model'};
for r = 1:size(rows, 1)
    w('<tr><td>%s</td>', rows{r, 1});
    for i = 1:numel(cols)
        if triage_col(i) && ~any(strcmp(rows{r, 2}, triage_rows))
            w('<td>-</td>');
        elseif triage_col(i) && strcmp(rows{r, 2}, 'false_delete')
            w('<td><b>%s</b> (real cells lost before the video)</td>', num_or_dash(cols{i}{2}.(rows{r, 2})));
        else
            w('<td>%s</td>', num_or_dash(cols{i}{2}.(rows{r, 2})));
        end
    end
    w('</tr>\n');
end
secs = results.seconds(results.seconds > 0);
w('<tr><td>median seconds per decision</td><td colspan="%d">%s</td></tr>\n', numel(cols), num_or_dash(median_or_nan(secs)));
w('<tr><td>items you flagged "I disagree with the key"</td><td colspan="%d">%d</td></tr>\n', numel(cols), numel(flag_cols));
if strcmp(score_basis, 'first')
    w('<tr><td>decisions you changed after the feedback (not scored)</td><td colspan="%d">%d</td></tr>\n', numel(cols), n_corrected);
end
w('</table>\n');
if key.has_motion_field ~= 1
    w('<p class="small">This session was reviewed before the m (motion) key existed, so motion rows are not available.</p>\n');
end

% --- error types ----------------------------------------------------------------------
w('<h2>What kind of items you missed</h2>\n<table><tr><th>hint category</th><th>false keeps</th><th>false deletes</th></tr>\n');
et = S.error_types;
legend = key.hint_legend;
for i = 1:size(et, 1)
    if et(i, 2) + et(i, 3) == 0; continue; end
    w('<tr><td>%s</td><td>%d</td><td>%d</td></tr>\n', esc(legend_text(legend, et(i, 1))), et(i, 2), et(i, 3));
end
w('</table>\n<p class="small">Hints are heuristics computed from the candidate''s features (within-session percentiles); they explain the reference decision, they do not define it.</p>\n');

% --- disagreements with panels ----------------------------------------------------------
dis_cols = [S.false_keep_cols(:); S.false_delete_cols(:)];
w('<h2>Where you disagreed with the reference (%d)</h2>\n', numel(dis_cols));
if isempty(dis_cols)
    w('<p>None.</p>\n');
end
n_drawn = 0;
if ~isempty(dis_cols)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1280, 560]);
    for i = 1:numel(dis_cols)
        col = dis_cols(i);
        is_fk = any(S.false_keep_cols == col);
        if is_fk; cls = 'fk'; kind = 'FALSE KEEP: you kept it, the reference deleted it';
        else; cls = 'fd'; kind = 'FALSE DELETE: you deleted it, the reference kept it'; end
        w('<div class="panel %s"><b>Neuron %d</b> (review column %d) -- %s<br>\n', cls, ctx.pos_of_col(col), col, kind);
        w('%s%s<br>\n', esc(decision_line(key, col, dec(col), mot(col))), esc(changed_note(col, changed_cols, dlast, mlast)));
        w('<span class="hint">%s</span>', esc(key.hint{col}));
        if any(flag_cols == col); w(' &nbsp; <b>[you flagged this one]</b>'); end
        w('<br>\n');
        if n_drawn < max_panels
            try
                partner = 0;
                if key.spatial_ok == 1 && key.hint_code(col) == 4 && key.overlap_partner(col) > 0 ...
                        && key.overlap_max(col) >= key.params.DUP_OVERLAP
                    partner = key.overlap_partner(col);
                end
                acorn_training_draw_panel(fig, neuron, ctx, col, ...
                    sprintf('Neuron %d', ctx.pos_of_col(col)), ...
                    struct('show_cn', true, 'partner_col', partner, ...
                           'title_color', ternary(is_fk, 'r', 'k')));
                png = fullfile(img_dir, sprintf('col%04d.png', col));
                print(fig, png, '-dpng', '-r80');
                w('<img src="%s/col%04d.png" alt="neuron %d">\n', esc([name, '_files']), col, ctx.pos_of_col(col));
                n_drawn = n_drawn + 1;
            catch err
                w('<p class="small">(panel could not be drawn: %s)</p>\n', esc(err.message));
            end
        else
            w('<p class="small">(image omitted: more than %d disagreements)</p>\n', max_panels);
        end
        w('</div>\n');
    end
    close(fig);
end

% --- contested ----------------------------------------------------------------------------
cc = S.contested_cols(:);
w('<h2>Contested items (reference and model disagree; not counted): %d</h2>\n', numel(cc));
if ~isempty(cc)
    w('<p>You sided with the reference on %d and with the model on %d of these %d. Uncounted, but worth watching across sessions.</p>\n', ...
        S.contested_with_ref, S.contested_with_model, S.n_contested);
    w('<table><tr><th>neuron</th><th>you</th><th>reference</th><th>model score</th><th>hint</th></tr>\n');
    for i = 1:numel(cc)
        col = cc(i);
        w('<tr><td>%d</td><td>%s%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n', ctx.pos_of_col(col), ...
            esc(you_word(dec(col), mot(col))), esc(changed_note(col, changed_cols, dlast, mlast)), esc(ref_word(key, col)), ...
            num_or_dash(key.model_score(col)), esc(key.hint{col}));
    end
    w('</table>\n');
end

% --- flagged ---------------------------------------------------------------------------
w('<h2>Items you flagged "I disagree with the key": %d</h2>\n', numel(flag_cols));
if ~isempty(flag_cols)
    w('<table><tr><th>neuron</th><th>you</th><th>reference</th><th>model score</th><th>hint</th></tr>\n');
    for i = 1:numel(flag_cols)
        col = flag_cols(i);
        w('<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n', ctx.pos_of_col(col), ...
            esc(you_word(dec(col), mot(col))), esc(ref_word(key, col)), ...
            num_or_dash(key.model_score(col)), esc(key.hint{col}));
    end
    w('</table>\n<p class="small">Tell the operator about these: a flagged item may be a mistake in the reference labels.</p>\n');
end

% --- unvisited -----------------------------------------------------------------------------
uv = S.unvisited_cols(:);
if ~isempty(uv)
    w('<h2>Never visited (%d, counted as keep like the real tool)</h2>\n<p>Neurons: %s</p>\n', numel(uv), ...
        esc(strjoin(arrayfun(@(c) sprintf('%d', ctx.pos_of_col(c)), uv, 'UniformOutput', false), ', ')));
end

% --- footer ---------------------------------------------------------------------------------
w('<hr><p class="small">Parameters (named assumptions, stored in the key): ');
pf = fieldnames(key.params);
for i = 1:numel(pf); w('%s=%g ', esc(pf{i}), key.params.(pf{i})); end
w('<br>Key: %s (built %s). Model scores are %s. Result file: %s. Repo %s.</p>\n', ...
    esc(key.path), esc(key.key_built_at), esc(key.score_kind), esc(results.result_file), esc(results.repo_hash));
w('</body></html>\n');
clear cleaner;

fprintf('report: %s\n', html);
if do_open && ~headless
    try
        web(html, '-browser');
    catch
        fprintf('open it in a browser: %s\n', html);
    end
end
end

% ------------------------------------------------------------------------------------------
function s = esc(x)
s = char(x);
s = regexprep(s, '[^\x09\x0A\x0D\x20-\x7E]', '?');
s = strrep(s, '&', '&amp;');
s = strrep(s, '<', '&lt;');
s = strrep(s, '>', '&gt;');
end

function s = num_or_dash(v)
if isempty(v) || (isnumeric(v) && isnan(v(1)))
    s = '-';
elseif isnumeric(v) && v(1) == round(v(1))
    s = sprintf('%d', v(1));
else
    s = sprintf('%.3f', v(1));
end
end

function m = median_or_nan(v)
if isempty(v); m = NaN; else; m = median(v); end
end

function s = you_word(d, mt)
if isnan(d); s = 'unvisited (keep)';
elseif d == 1; s = 'keep';
elseif mt == 1; s = 'delete (m)';
else; s = 'delete'; end
end

function s = ref_word(key, col)
if key.ref_keep(col) == 1; s = 'keep'; else; s = 'delete'; end
if key.ref_motion(col) == 1; s = [s, ' (motion)']; end
end

function s = decision_line(key, col, d, mt)
if key.model_available && ~isnan(key.model_score(col)); ms = sprintf('%.2f', key.model_score(col)); else; ms = 'n/a'; end
s = sprintf('You: %s | Reference: %s | Model score: %s', you_word(d, mt), ref_word(key, col), ms);
end

function s = changed_note(col, changed_cols, dlast, mlast)
s = '';
if any(changed_cols == col)
    s = sprintf(' (changed to %s after the feedback; not scored)', you_word(dlast(col), mlast(col)));
end
end

function s = legend_text(legend, code)
s = sprintf('%d', code);
for i = 1:numel(legend)
    if startsWith(legend{i}, sprintf('%d:', code))
        s = legend{i};
        return;
    end
end
end

function v = ternary(c, a, b)
if c; v = a; else; v = b; end
end

function v = getopt(o, name, default)
if isfield(o, name) && ~isempty(o.(name))
    v = o.(name);
else
    v = default;
end
end
