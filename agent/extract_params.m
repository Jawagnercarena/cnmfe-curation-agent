%% extract_params.m
% One-time setup script. Reads gSig and gSiz from every completed session's
% neuron.mat and writes agent/config/animal_params.json.
% Run this from the CNMF_E_LEGACY_BIANE_CLAUDE root directory.
%
% Flags animals where gSig or gSiz is inconsistent across sessions so you
% can decide which value to use (or fall back to the safe default of gSiz=36).

clear; clc;

%% --- Configuration ---
% Root folder containing one subdirectory per brain area (BLA, vCA1, DG, ...).
% All areas are scanned automatically -- no manual update needed when adding areas.
base_root   = getenv('CNMFE_DATA_PARENT');
if isempty(base_root); base_root = 'D:\Julian_CNMFe'; end
output_json = fullfile(fileparts(mfilename('fullpath')), 'config', 'animal_params.json');

% Add Sources2D class to path so neuron.mat loads correctly
script_dir = fileparts(mfilename('fullpath'));
repo_root  = fileparts(script_dir);  % one level up from agent/
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));

%% --- Walk every area / task / session folder ---
area_dirs = dir(base_root);
area_dirs = area_dirs([area_dirs.isdir] & ~startsWith({area_dirs.name}, '.'));

records = {};  % {animal_id, area, task, session, gSig, gSiz}

fprintf('Scanning sessions in %s ...\n\n', base_root);

for ar = 1:numel(area_dirs)
    area_name = area_dirs(ar).name;
    area_path = fullfile(base_root, area_name);

    task_dirs = dir(area_path);
    task_dirs = task_dirs([task_dirs.isdir] & ~startsWith({task_dirs.name}, '.'));
    if isempty(task_dirs), continue; end

    fprintf('[%s]\n', area_name);

    for t = 1:numel(task_dirs)
        task_path = fullfile(area_path, task_dirs(t).name);
        session_dirs = dir(task_path);
        session_dirs = session_dirs([session_dirs.isdir] & ~startsWith({session_dirs.name}, '.'));

        for s = 1:numel(session_dirs)
            session_name = session_dirs(s).name;
            session_path = fullfile(task_path, session_name);
            neuron_file  = fullfile(session_path, 'neuron.mat');

            if ~isfile(neuron_file)
                continue;  % unprocessed session -- skip
            end

            % Parse animal ID: format is AVGNx-TSeries-DATE-ANIMALID-...
            % Skip any folder not starting with AVG (e.g. error-AVG..., temp-, etc.)
            if ~strncmpi(session_name, 'AVG', 3)
                fprintf('  [SKIP] Non-standard folder name (not AVG prefix): %s\n', session_name);
                continue;
            end
            parts = strsplit(session_name, '-');
            if numel(parts) < 4
                fprintf('  [SKIP] Cannot parse animal ID from: %s\n', session_name);
                continue;
            end
            animal_id = parts{4};  % e.g. 'bla8', '94', '921'

            % Load neuron object
            try
                data_loaded = load(neuron_file, 'neuron');
                neuron = data_loaded.neuron;
                gSig = neuron.options.gSig;
                gSiz = neuron.options.gSiz;
            catch err
                fprintf('  [ERROR] %s/%s/%s: %s\n', area_name, task_dirs(t).name, session_name, err.message);
                continue;
            end

            fprintf('  %s/%s  ->  animal=%s  gSig=%d  gSiz=%d\n', ...
                task_dirs(t).name, session_name, animal_id, gSig, gSiz);

            records{end+1} = struct( ...
                'animal',  animal_id, ...
                'area',    area_name, ...
                'task',    task_dirs(t).name, ...
                'session', session_name, ...
                'gSig',    gSig, ...
                'gSiz',    gSiz);
        end
    end
end

fprintf('\nFound %d completed sessions.\n\n', numel(records));

if isempty(records)
    error('No completed sessions found. Check base_root path.');
end

%% --- Build per-animal summary, flag inconsistencies ---
animals = unique(cellfun(@(r) r.animal, records, 'UniformOutput', false));

animal_params = struct();
warnings_found = false;

for a = 1:numel(animals)
    aid = animals{a};
    % MATLAB struct field names cannot start with a digit.
    % Prefix numeric IDs with 'x' for struct storage; stripped from JSON below.
    if ~isempty(regexp(aid, '^\d', 'once'))
        struct_key = ['x' aid];
    else
        struct_key = aid;
    end
    mask = cellfun(@(r) strcmp(r.animal, aid), records);
    subset = records(mask);

    gSig_vals = cellfun(@(r) r.gSig, subset);
    gSiz_vals = cellfun(@(r) r.gSiz, subset);

    consistent = (numel(unique(gSig_vals)) == 1) && (numel(unique(gSiz_vals)) == 1);

    if consistent
        chosen_gSig = gSig_vals(1);
        chosen_gSiz = gSiz_vals(1);
        flag = 'ok';
    else
        % Inconsistent: use the most common value; if tied, use the larger
        gSiz_mode = mode(gSiz_vals);
        gSig_mode = mode(gSig_vals);
        % If mode is tied, max() is the safer choice per user preference
        if sum(gSiz_vals == gSiz_mode) == 1
            chosen_gSiz = max(gSiz_vals);
            chosen_gSig = max(gSig_vals);
        else
            chosen_gSiz = gSiz_mode;
            chosen_gSig = gSig_mode;
        end
        flag = 'inconsistent_check_manually';
        warnings_found = true;

        fprintf('  [WARNING] Animal %s has inconsistent gSig/gSiz across sessions:\n', aid);
        for i = 1:numel(subset)
            fprintf('    %s/%s: gSig=%d gSiz=%d\n', ...
                subset{i}.task, subset{i}.session, gSig_vals(i), gSiz_vals(i));
        end
        fprintf('    Using gSig=%d gSiz=%d (most common / larger). Edit JSON to override.\n\n', ...
            chosen_gSig, chosen_gSiz);
    end

    animal_params.(struct_key) = struct( ...
        'area',       subset{1}.area, ...
        'gSig',       chosen_gSig, ...
        'gSiz',       chosen_gSiz, ...
        'flag',       flag, ...
        'n_sessions', numel(subset), ...
        'gSig_all',   gSig_vals, ...
        'gSiz_all',   gSiz_vals);
end

%% --- Write JSON ---
json_str = jsonencode(animal_params, 'PrettyPrint', true);
% Strip the 'x' prefix added to numeric animal IDs (MATLAB struct field name constraint).
% e.g. "x94": -> "94":  and  "x100": -> "100":
json_str = regexprep(json_str, '"x(\d+)":', '"$1":');
fid = fopen(output_json, 'w');
if fid == -1
    error('Cannot write to %s', output_json);
end
fwrite(fid, json_str, 'char');
fclose(fid);

fprintf('\n=== animal_params.json written to:\n    %s\n\n', output_json);

%% --- Print summary table ---
fprintf('%-12s  %5s  %5s  %10s  %s\n', 'Animal', 'gSig', 'gSiz', 'Sessions', 'Flag');
fprintf('%s\n', repmat('-', 1, 55));
for a = 1:numel(animals)
    aid = animals{a};
    if ~isempty(regexp(aid, '^\d', 'once'))
        sk = ['x' aid];
    else
        sk = aid;
    end
    p   = animal_params.(sk);
    fprintf('%-12s  %5d  %5d  %10d  %s\n', aid, p.gSig, p.gSiz, p.n_sessions, p.flag);
end

if warnings_found
    fprintf('\n[!] Some animals had inconsistent parameters. Review the warnings above\n');
    fprintf('    and edit animal_params.json manually if needed.\n');
else
    fprintf('\nAll animals have consistent gSig/gSiz across sessions.\n');
end

fprintf('\nDone.\n');
