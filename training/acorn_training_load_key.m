function key = acorn_training_load_key(session_dir)
% acorn_training_load_key -- load and validate training_key.mat.
%
%   key = acorn_training_load_key(session_dir)
%
% The key is written by agent/build_training_key.py (scipy savemat) and holds,
% per review candidate (one per column of review_neuron.mat, in that order),
% the reference reviewer's decision, the deployed model's score, the contested
% flag, the feature row + within-session ranks and a plain-language hint.  This
% loader casts every numeric vector to a double column, keeps strings as char
% and cells as cells, fills defaults for optional fields and checks lengths.
%
% Adds: key.model_available (logical), key.path (char).

kp = fullfile(session_dir, 'training_key.mat');
if ~isfile(kp)
    error(['training_key.mat not found in %s. The central operator builds it with ' ...
           'agent/build_training_key.py and push_training_bundle.py ships it.'], session_dir);
end
key = load(kp);
key.path = kp;

scalars = {'schema_version', 'n_candidates', 'n_review', 'n_auto_rejected', ...
           'has_motion_field', 'model_n_sessions', 'model_n_features', ...
           'model_feature_version', 'model_reject_threshold', 'n_features', ...
           'ranks_match_npz', 'spatial_ok'};
for i = 1:numel(scalars)
    f = scalars{i};
    if isfield(key, f) && ~isempty(key.(f))
        key.(f) = double(key.(f)(1));
    else
        key.(f) = 0;
    end
end

n = key.n_review;
vectors = {'review_col', 'npz_row0', 'ref_keep', 'ref_motion', 'model_score', ...
           'model_keep', 'contested', 'clear_cut', 'hint_code', 'overlap_max', ...
           'overlap_partner', 'n_pixels'};
for i = 1:numel(vectors)
    f = vectors{i};
    if isfield(key, f)
        key.(f) = double(key.(f)(:));
        if numel(key.(f)) ~= n
            error('training_key.mat: %s has %d entries, expected n_review=%d', f, numel(key.(f)), n);
        end
    end
end
if ~isfield(key, 'overlap_max');     key.overlap_max = zeros(n, 1);     end
if ~isfield(key, 'overlap_partner'); key.overlap_partner = zeros(n, 1); end
if ~isfield(key, 'n_pixels');        key.n_pixels = zeros(n, 1);        end
if ~isfield(key, 'centroid_yx');     key.centroid_yx = zeros(n, 2);     end
key.centroid_yx = double(key.centroid_yx);

strings = {'area', 'task', 'session', 'reference_reviewer', 'key_built_at', ...
           'score_kind', 'model_path', 'model_type', 'augmented_at'};
for i = 1:numel(strings)
    f = strings{i};
    if isfield(key, f)
        key.(f) = strtrim(char(key.(f)));
        if strcmp(key.(f), '-'); key.(f) = ''; end
    else
        key.(f) = '';
    end
end

if ~isfield(key, 'feature_names') || ~iscell(key.feature_names)
    error('training_key.mat: feature_names missing or not a cell');
end
key.feature_names = cellfun(@(s) strtrim(char(s)), key.feature_names(:)', 'UniformOutput', false);
key.features = double(key.features);
key.ranks = double(key.ranks);
if size(key.features, 1) ~= n || size(key.features, 2) ~= numel(key.feature_names)
    error('training_key.mat: features is %dx%d, expected %dx%d', ...
        size(key.features, 1), size(key.features, 2), n, numel(key.feature_names));
end
if ~isfield(key, 'hint') || ~iscell(key.hint)
    error('training_key.mat: hint missing or not a cell');
end
key.hint = cellfun(@(s) strtrim(char(s)), key.hint(:), 'UniformOutput', false);
if ~isfield(key, 'hint_legend') || ~iscell(key.hint_legend)
    key.hint_legend = {};
end
key.hint_legend = cellfun(@(s) strtrim(char(s)), key.hint_legend(:), 'UniformOutput', false);

if ~isfield(key, 'params') || ~isstruct(key.params)
    error('training_key.mat: params struct missing');
end
pf = fieldnames(key.params);
for i = 1:numel(pf)
    key.params.(pf{i}) = double(key.params.(pf{i})(1));
end

key.model_available = strcmp(key.score_kind, 'insample') || strcmp(key.score_kind, 'oof');
if ~key.model_available
    key.contested = zeros(n, 1);
    key.clear_cut = zeros(n, 1);
end
end
