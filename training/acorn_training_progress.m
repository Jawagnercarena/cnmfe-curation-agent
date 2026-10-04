function html = acorn_training_progress(trainee, opts)
% acorn_training_progress -- plot one trainee's attempts over time.
%
%   html = acorn_training_progress(trainee, opts)
%
% Reads the machine-level progress.csv (acorn_training_paths) -- or
% opts.progress_csv -- and writes progress.html + progress.png next to it:
% agreement, kappa, false-keep rate and false-delete rate per attempt, in
% time order, plus a table of every attempt.  One point per attempt (the
% 'final' row when a video pass was done, else pass 1).  Numbers only: the
% supervisor decides what they mean.
% opts.headless suppresses the browser; opts.root_override (tests).

if nargin < 2 || isempty(opts); opts = struct(); end
headless = logical(getopt(opts, 'headless', false));
root_override = getopt(opts, 'root_override', '');
P = acorn_training_paths(trainee, root_override);
csv = getopt(opts, 'progress_csv', P.progress_csv);
out_dir = getopt(opts, 'out_dir', fileparts(csv));
html = '';
if ~isfile(csv)
    fprintf('No progress file yet at %s -- do a drill first.\n', csv);
    return;
end

cols = acorn_training_progress_columns();
io = detectImportOptions(csv, 'Delimiter', ',', 'TextType', 'char');
io.VariableNamingRule = 'preserve';
charcols = {'timestamp', 'trainee', 'area', 'task', 'session', 'mode', 'pass', 'subset', 'result_file', 'repo_hash'};
io = setvartype(io, intersect(charcols, io.VariableNames), 'char');
io = setvartype(io, setdiff(io.VariableNames, charcols), 'double');
T = readtable(csv, io);
if ~isequal(T.Properties.VariableNames, cols)
    fprintf(2, 'progress.csv header differs from the expected columns; plotting what can be read.\n');
end
keep = strcmpi(strtrim(T.trainee), strtrim(P.trainee)) | strcmpi(strtrim(T.trainee), strtrim(trainee));
T = T(keep, :);
if isempty(T)
    fprintf('No attempts recorded for %s in %s.\n', trainee, csv);
    return;
end
% one row per attempt: 'final' if present for that result file, else pass 1
[~, ~, g] = unique(T.result_file, 'stable');
pick = false(height(T), 1);
for k = 1:max(g)
    idx = find(g == k);
    f = idx(strcmp(T.pass(idx), 'final'));
    if isempty(f); f = idx(strcmp(T.pass(idx), '1')); end
    if isempty(f); f = idx(1); end
    pick(f(1)) = true;
end
T = T(pick, :);
[~, order] = sort(T.timestamp);
T = T(order, :);
k = height(T);
x = (1:k)';
fk_rate = T.false_keep ./ max(1, T.n_ref_delete_scored);
fd_rate = T.false_delete ./ max(1, T.n_ref_keep_scored);

fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 700]);
subplot(2, 1, 1);
plot(x, T.agreement, '-o', 'LineWidth', 1.5); hold on;
plot(x, T.kappa, '-s', 'LineWidth', 1.5);
ylim([-0.05, 1.05]); grid on;
xlabel('attempt (time order)'); ylabel('value');
legend({'agreement', 'kappa'}, 'Location', 'southeast');
title(sprintf('%s: agreement with the reference across %d attempt(s)', trainee, k), 'Interpreter', 'none');
subplot(2, 1, 2);
plot(x, fk_rate, '-o', 'LineWidth', 1.5); hold on;
plot(x, fd_rate, '-s', 'LineWidth', 1.5);
ylim([-0.05, 1.05]); grid on;
xlabel('attempt (time order)'); ylabel('rate');
legend({'false keep rate (kept / reference deletes)', 'false delete rate (deleted / reference keeps)'}, 'Location', 'northeast');
set(gca, 'XTick', x);
labels = cell(k, 1);
for i = 1:k
    labels{i} = sprintf('%d', i);
end
set(gca, 'XTickLabel', labels);
png = fullfile(out_dir, 'progress.png');
print(fig, png, '-dpng', '-r90');
close(fig);

html = fullfile(out_dir, 'progress.html');
fid = fopen(html, 'w');
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '<!DOCTYPE html>\n<html><head><meta charset="ascii"><title>ACORN training progress %s</title>\n', esc(trainee));
fprintf(fid, '<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;color:#222}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 7px;font-size:12px}th{background:#eee}</style></head><body>\n');
fprintf(fid, '<h1>ACORN training progress: %s</h1>\n<img src="progress.png" style="max-width:100%%">\n', esc(trainee));
fprintf(fid, '<h2>Attempts (%d)</h2>\n<table><tr><th>#</th><th>finished</th><th>area/task/session</th><th>stage</th><th>mode</th><th>pass</th><th>subset</th><th>shown</th><th>scored</th><th>contested</th><th>agreement</th><th>kappa</th><th>false keeps</th><th>false deletes</th><th>m-tags on motion</th><th>median s</th><th>flagged</th></tr>\n', k);
for i = 1:k
    fprintf(fid, '<tr><td>%d</td><td>%s</td><td>%s / %s / %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s / %s</td><td>%s / %s</td><td>%s / %s</td><td>%s</td><td>%s</td></tr>\n', ...
        i, esc(T.timestamp{i}), esc(T.area{i}), esc(T.task{i}), esc(T.session{i}), nd(T.stage(i)), ...
        esc(T.mode{i}), esc(T.pass{i}), esc(T.subset{i}), nd(T.n_items(i)), nd(T.n_scored(i)), ...
        nd(T.n_contested(i)), nd(T.agreement(i)), nd(T.kappa(i)), nd(T.false_keep(i)), ...
        nd(T.n_ref_delete_scored(i)), nd(T.false_delete(i)), nd(T.n_ref_keep_scored(i)), ...
        nd(T.motion_tagged_m(i)), nd(T.motion_ref_n(i)), nd(T.median_sec(i)), nd(T.n_flagged(i)));
end
fprintf(fid, '</table>\n<p style="color:#666;font-size:12px">Source: %s</p></body></html>\n', esc(csv));
clear cleaner;
fprintf('progress: %s\n', html);
if ~headless
    try; web(html, '-browser'); catch; end
end
end

function s = esc(x)
s = char(x);
s = regexprep(s, '[^\x09\x0A\x0D\x20-\x7E]', '?');
s = strrep(s, '&', '&amp;'); s = strrep(s, '<', '&lt;'); s = strrep(s, '>', '&gt;');
end

function s = nd(v)
if isempty(v) || isnan(v(1)); s = '-';
elseif v(1) == round(v(1)); s = sprintf('%d', v(1));
else; s = sprintf('%.3f', v(1)); end
end

function v = getopt(o, name, default)
if isfield(o, name) && ~isempty(o.(name)); v = o.(name); else; v = default; end
end
