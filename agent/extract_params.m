%% extract_params.m
% Reads gSig and gSiz from every completed session's neuron.mat and writes
% agent/config/animal_params.json. The watchers run it whenever a neuron.mat
% is newer than that JSON (watcher.py _refresh_animal_params); it can also be
% run by hand from the CNMF_E_LEGACY_BIANE_CLAUDE root directory.
%
% Flags animals where gSig or gSiz is inconsistent across sessions so you
% can decide which value to use (or fall back to the safe default of gSiz=36).
%
% Per-session cache (2026-11-07): loading every neuron.mat in every area on
% each refresh (446 files, 6.8 GB) outran the watcher's 15-min timeout while
% D: was busy, and every timed-out attempt started over. gSig/gSiz are now
% cached per session in <data parent>/.params_cache/extract_params_cache.mat,
% keyed by area/task/session and reused while that neuron.mat keeps the same
% byte size and modification time, so only new or changed files are loaded.
% The cache is saved after every newly loaded file, so a run that is killed
% keeps its progress for the next attempt. Delete the cache file to force a
% full re-read. The JSON written is the same as without the cache.
%
% Environment overrides: CNMFE_DATA_PARENT (data root) and
% CNMFE_ANIMAL_PARAMS_JSON (output file; used to test against scratch paths).

clear; clc;

%% --- Configuration ---
% Root folder containing one subdirectory per brain area (BLA, vCA1, DG, ...).
% All areas are scanned automatically -- no manual update needed when adding areas.
base_root   = getenv('CNMFE_DATA_PARENT');
if isempty(base_root); base_root = 'D:\Julian_CNMFe'; end
output_json = getenv('CNMFE_ANIMAL_PARAMS_JSON');
if isempty(output_json)
    output_json = fullfile(fileparts(mfilename('fullpath')), 'config', 'animal_params.json');
end
cache_dir  = fullfile(base_root, '.params_cache');
cache_file = fullfile(cache_dir, 'extract_params_cache.mat');

% Add Sources2D class to path so neuron.mat loads correctly
script_dir = fileparts(mfilename('fullpath'));
repo_root  = fileparts(script_dir);  % one level up from agent/
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));

%% --- Per-session cache ---
params_cache = load_params_cache(cache_file);
fprintf('Cache: %d session(s) in %s\n', params_cache.Count, cache_file);
n_loaded  = 0;
n_cached  = 0;
seen_keys = {};

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

            % gSig/gSiz from the cache when this neuron.mat is unchanged,
            % otherwise from the neuron object itself.
            neuron_info = dir(neuron_file);
            cache_key   = [area_name '/' task_dirs(t).name '/' session_name];
            [gSig, gSiz, hit] = cached_params(params_cache, cache_key, neuron_info);
            if hit
                n_cached = n_cached + 1;
            else
                try
                    data_loaded = load(neuron_file, 'neuron');
                    neuron = data_loaded.neuron;
                    gSig = neuron.options.gSig;
                    gSiz = neuron.options.gSiz;
                catch err
                    fprintf('  [ERROR] %s/%s/%s: %s\n', area_name, task_dirs(t).name, session_name, err.message);
                    continue;
                end
                n_loaded = n_loaded + 1;
                params_cache(cache_key) = struct('key', cache_key, ...
                    'bytes', neuron_info.bytes, 'datenum', neuron_info.datenum, ...
                    'gSig', gSig, 'gSiz', gSiz);
                save_params_cache(cache_dir, cache_file, params_cache, {}, false);
            end
            seen_keys{end+1} = cache_key; %#ok<SAGROW>

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

fprintf('\nFound %d completed sessions (%d read from neuron.mat, %d from cache).\n\n', ...
    numel(records), n_loaded, n_cached);

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

% Drop cache entries for sessions that no longer exist or failed to load.
save_params_cache(cache_dir, cache_file, params_cache, seen_keys, true);

fprintf('\nDone.\n');


%% --- Local functions ---

function params_cache = load_params_cache(cache_file)
% Map from 'area/task/session' to struct(key, bytes, datenum, gSig, gSiz).
% A missing or unreadable cache gives an empty map: every neuron.mat is read.
params_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
if ~isfile(cache_file)
    return;
end
try
    S = load(cache_file, 'entries');
    entries = S.entries;
    need = {'key', 'bytes', 'datenum', 'gSig', 'gSiz'};
    if ~isempty(entries) && ~(isstruct(entries) && all(isfield(entries, need)))
        error('cache entries do not have the fields %s', strjoin(need, ', '));
    end
    for i = 1:numel(entries)
        params_cache(entries(i).key) = entries(i);
    end
catch err
    fprintf('[WARN] Ignoring unreadable cache %s (%s); reading every neuron.mat.\n', ...
        cache_file, err.message);
    params_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
end
end

function [gSig, gSiz, hit] = cached_params(params_cache, cache_key, neuron_info)
% Cached values only if the neuron.mat has the same size and modification
% time as when they were read (datenum is in days; 1e-5 day is under 1 s).
gSig = [];
gSiz = [];
hit  = false;
if ~isKey(params_cache, cache_key)
    return;
end
c = params_cache(cache_key);
if c.bytes == neuron_info.bytes && abs(c.datenum - neuron_info.datenum) < 1e-5
    gSig = c.gSig;
    gSiz = c.gSiz;
    hit  = true;
end
end

function save_params_cache(cache_dir, cache_file, params_cache, keep_keys, prune)
% Write all entries (prune=false) or only those in keep_keys (prune=true).
% Saved to a temp file and renamed over the cache, so a run killed mid-write
% leaves the previous cache intact. A failed write only costs speed.
try
    keys_out = params_cache.keys;
    if prune
        keys_out = intersect(keys_out, keep_keys);
    end
    if isempty(keys_out)
        entries = struct('key', {}, 'bytes', {}, 'datenum', {}, 'gSig', {}, 'gSiz', {});
    else
        vals = params_cache.values(keys_out);
        entries = [vals{:}];
    end
    if ~isfolder(cache_dir)
        mkdir(cache_dir);
    end
    tmp_file = fullfile(cache_dir, 'extract_params_cache_tmp.mat');
    save(tmp_file, 'entries');
    movefile(tmp_file, cache_file, 'f');
catch err
    fprintf('[WARN] Could not save the params cache %s: %s\n', cache_file, err.message);
end
end
