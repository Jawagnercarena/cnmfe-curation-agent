function acorn_training_append_progress(csv_path, row)
% acorn_training_append_progress -- append one attempt to a progress.csv.
%
%   acorn_training_append_progress(csv_path, row)
%
% row is a struct with (any of) the fields below; missing fields are written
% empty.  The column order is the single definition shared with
% agent/training_common.py (PROGRESS_COLUMNS); summarize_training.py refuses
% files whose header differs.  Header written when the file is new.  ASCII, LF.

cols = acorn_training_progress_columns();
folder = fileparts(csv_path);
if ~isempty(folder) && ~exist(folder, 'dir'); mkdir(folder); end
is_new = ~isfile(csv_path) || dir(csv_path).bytes == 0;
if ~is_new
    migrate_header(csv_path, cols);
end
fid = fopen(csv_path, 'a');
if fid < 0; error('acorn_training_append_progress: cannot open %s', csv_path); end
try
    if is_new
        fprintf(fid, '%s\n', strjoin(cols, ','));
    end
    vals = cell(1, numel(cols));
    for i = 1:numel(cols)
        if isfield(row, cols{i})
            vals{i} = fmt(row.(cols{i}));
        else
            vals{i} = '';
        end
    end
    fprintf(fid, '%s\n', strjoin(vals, ','));
catch err
    fclose(fid);
    rethrow(err);
end
fclose(fid);
end

function migrate_header(csv_path, cols)
% An older progress.csv (fewer trailing columns) is extended in place: the
% header becomes the current one and every existing row is padded with empty
% fields, so nothing is lost and no column shifts.  Any other header is an
% error -- never append rows to a file whose columns are unknown.
txt = fileread(csv_path);
lines = regexp(txt, '\r?\n', 'split');
if isempty(lines) || isempty(strtrim(lines{1})); return; end
want = strjoin(cols, ',');
old = strtrim(lines{1});
if strcmp(old, want); return; end
oldcols = strsplit(old, ',');
if numel(oldcols) < numel(cols) && isequal(oldcols, cols(1:numel(oldcols)))
    pad = repmat(',', 1, numel(cols) - numel(oldcols));
    out = {want};
    for i = 2:numel(lines)
        if isempty(strtrim(lines{i})); continue; end
        out{end+1} = [strtrim(lines{i}), pad]; %#ok<AGROW>
    end
    fid = fopen(csv_path, 'w');
    if fid < 0; error('acorn_training_append_progress: cannot rewrite %s', csv_path); end
    fprintf(fid, '%s\n', out{:});
    fclose(fid);
    fprintf('progress.csv header extended from %d to %d columns (%d older row(s) padded): %s\n', ...
        numel(oldcols), numel(cols), numel(out) - 1, csv_path);
else
    error(['acorn_training_append_progress: %s has an unexpected header; refusing to append. ' ...
           'Move the file aside if it is not a training progress file.'], csv_path);
end
end

function s = fmt(v)
if ischar(v) || isstring(v)
    s = char(v);
    s = regexprep(s, '[,\r\n]+', ';');
    s = regexprep(s, '[^\x20-\x7E]', '?');
elseif isempty(v)
    s = '';
elseif islogical(v)
    s = sprintf('%d', double(v(1)));
elseif isnumeric(v)
    v = double(v(1));
    if isnan(v)
        s = 'NaN';
    elseif v == round(v) && abs(v) < 1e12
        s = sprintf('%d', v);
    else
        s = sprintf('%.4f', v);
    end
else
    s = '';
end
end
