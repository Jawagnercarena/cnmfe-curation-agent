function acorn_training(session_dir)
% acorn_training -- menu for one training session (called by run_training.m).
%
%   acorn_training(session_dir)     (default: the current folder)
%
% The folder is a training bundle pushed by the central operator: it holds
% review_neuron.mat, training_key.mat, TRAINING_SESSION.txt, the raw
% {session}.mat video, optionally Ybg_weights.mat, and this launcher.
% Everything you produce goes to <session>/training_results/ -- copy that
% folder back to training/<your name>/returns/<session>/ on the exchange.
% Never copy a training folder into inbox/.  See docs/TRAINING.md.

if nargin < 1 || isempty(session_dir); session_dir = pwd; end
session_dir = char(session_dir);
while numel(session_dir) > 1 && (session_dir(end) == '\' || session_dir(end) == '/')
    session_dir = session_dir(1:end-1);
end
fprintf('\n==============================================================\n');
fprintf(' ACORN reviewer training\n');
fprintf('==============================================================\n');
marker = read_marker(fullfile(session_dir, 'TRAINING_SESSION.txt'));
key = acorn_training_load_key(session_dir);
has_video = isfile(fullfile(session_dir, [key.session, '.mat']));
fprintf('Session: %s / %s / %s\n', key.area, key.task, key.session);
fprintf('%d candidates to review. Reference reviewer: %s. Model scores: %s.\n', ...
    key.n_review, key.reference_reviewer, key.score_kind);
if ~has_video
    fprintf(2, ['No raw video in this bundle: only the static warm-up drill works here. ' ...
                'Ask the operator for a bundle with the video.\n']);
end
stage_hint = NaN;
if isfield(marker, 'stage_hint')
    stage_hint = str2double(marker.stage_hint);
end
if ~isnan(stage_hint); fprintf('Suggested stage for this bundle: %d\n', stage_hint); end

% --- trainee name ---------------------------------------------------------------------
default_name = '';
if isfield(marker, 'trainee') && ~isempty(marker.trainee); default_name = marker.trainee; end
P0 = acorn_training_paths('trainee');
if isempty(default_name) && isfile(P0.trainee_txt)
    default_name = strtrim(fileread(P0.trainee_txt));
end
name = '';
while isempty(name)
    if isempty(default_name)
        name = strtrim(input('Your name: ', 's'));
    else
        name = strtrim(input(sprintf('Your name [%s]: ', default_name), 's'));
        if isempty(name); name = default_name; end
    end
end
P = acorn_training_paths(name);
try
    fid = fopen(P.trainee_txt, 'w'); fprintf(fid, '%s\n', name); fclose(fid);
catch
end
fprintf('Progress file: %s\n', P.progress_csv);

% --- menu --------------------------------------------------------------------------------
while true
    fprintf('\n  1  Video drill: every candidate in the movie panel          [stages 1-2]\n');
    fprintf('  2  Workflow drill: static triage, then video on your keeps   [stage 3]\n');
    fprintf('  3  Dress rehearsal: the real review tool                     [stage 4]\n');
    fprintf('  4  My progress\n');
    fprintf('  5  Open the example gallery\n');
    fprintf('  6  Static warm-up drill (footprint + trace only, no video)\n');
    fprintf('  0  Quit\n');
    default_choice = '1';
    if stage_hint == 3; default_choice = '2'; elseif stage_hint == 4; default_choice = '3'; end
    if ~has_video && any(strcmp(default_choice, {'1', '2', '3'})); default_choice = '6'; end
    choice = strtrim(input(sprintf('Choose [%s]: ', default_choice), 's'));
    if isempty(choice); choice = default_choice; end
    switch choice
        case {'1', '2', '6'}
            if ~has_video && ~strcmp(choice, '6')
                fprintf('This bundle has no raw video; only the static warm-up (6) is available.\n');
                continue;
            end
            opts = struct('trainee', name, 'stage', stage_hint, 'progress_csv', P.progress_csv);
            if strcmp(choice, '1'); opts.drill_mode = 'video';
            elseif strcmp(choice, '2'); opts.drill_mode = 'workflow';
            else; opts.drill_mode = 'static'; end
            if stage_hint == 3; dm = 'exam'; else; dm = 'coach'; end
            m = strtrim(input(sprintf('Feedback: coach (after each decision) or exam (at the end) [%s]: ', dm), 's'));
            if isempty(m); m = dm; end
            opts.mode = lower(m);
            if stage_hint == 1; ds = 'clear'; else; ds = 'all'; end
            s = strtrim(input(sprintf('Items: all, or clear (clear-cut ones only) [%s]: ', ds), 's'));
            if isempty(s); s = ds; end
            opts.subset = lower(s);
            if ~strcmp(opts.drill_mode, 'video')
                c = strtrim(input('Show the correlation-image panel in the static view? y/n [y]: ', 's'));
                opts.show_cn = ~strcmpi(c, 'n');
            end
            try
                acorn_training_drill(session_dir, key, opts);
            catch err
                fprintf(2, 'Drill stopped: %s\n', err.message);
            end
        case '3'
            if ~has_video
                fprintf('The dress rehearsal needs the raw video; this bundle has none.\n');
                continue;
            end
            c = strtrim(input('This runs the real review tool (about 20 min before the first prompt). Continue? y/n [n]: ', 's'));
            if ~strcmpi(c, 'y'); continue; end
            try
                acorn_training_rehearsal(session_dir, key, struct('trainee', name, 'stage', 4, 'progress_csv', P.progress_csv));
            catch err
                fprintf(2, 'Rehearsal stopped: %s\n', err.message);
            end
        case '4'
            try
                acorn_training_progress(name);
            catch err
                fprintf(2, 'Progress view failed: %s\n', err.message);
            end
        case '5'
            g = '';
            if isfield(marker, 'gallery'); g = marker.gallery; end
            if ~isempty(g) && isfile(g)
                try; web(g, '-browser'); catch; fprintf('Open in a browser: %s\n', g); end
            else
                fprintf('Gallery not found (%s). Ask the operator for the path to _shared/gallery/index.html.\n', g);
            end
        case '0'
            fprintf('Bye. Remember to copy training_results/ back to training/%s/returns/%s/ on the exchange.\n', name, key.session);
            return;
        otherwise
            fprintf('Unknown choice.\n');
    end
end
end

function info = read_marker(path)
info = struct();
if ~isfile(path); return; end
txt = fileread(path);
lines = strsplit(txt, {'\r', '\n'});
for i = 1:numel(lines)
    ln = strtrim(lines{i});
    if isempty(ln) || startsWith(ln, 'NOTE:'); continue; end
    k = strfind(ln, ':');
    if isempty(k); continue; end
    f = regexprep(strtrim(ln(1:k(1)-1)), '[^A-Za-z0-9_]', '_');
    if isempty(f) || ~isletter(f(1)); continue; end
    info.(f) = strtrim(ln(k(1)+1:end));
end
end
