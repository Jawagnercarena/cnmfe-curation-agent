function acorn_training_key_augment(manifest_txt)
% acorn_training_key_augment -- add spatial-overlap fields to training keys.
%
% Central-machine step, launched headless by agent/build_training_key.py.
% The manifest is a text file with one line per key:
%     <key_path><TAB><session_dir>
% For each line it loads <session_dir>/review_neuron.mat (the candidate set
% the reviewer saw), computes the cosine overlap between every pair of
% candidate footprints (same normalisation CNMFe_final_save.m uses to derive
% labels), and writes <key_dir>/training_key_spatial.mat with
%     overlap_max      n x 1   best overlap with any OTHER candidate
%     overlap_partner  n x 1   1-based review column of that candidate (0 if n==1)
%     centroid_yx      n x 2   footprint centre of mass (row, col)
%     n_pixels         n x 1   non-zero pixels in the footprint
%     spatial_ok       1 if size(neuron.A,2) matched the key's n_review
%     augmented_at     char timestamp
% Python merges that file into training_key.mat.  Nothing in the session
% folder is written.  Errors on one session are reported and do not stop the
% others.

fid = fopen(manifest_txt, 'r');
if fid < 0
    error('acorn_training_key_augment: cannot open manifest %s', manifest_txt);
end
lines = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    ln = strtrim(ln);
    if ~isempty(ln); lines{end+1} = ln; end %#ok<AGROW>
end
fclose(fid);
fprintf('acorn_training_key_augment: %d key(s)\n', numel(lines));

n_ok = 0; n_fail = 0;
for i = 1:numel(lines)
    parts = strsplit(lines{i}, sprintf('\t'));
    if numel(parts) ~= 2
        fprintf(2, '  bad manifest line %d: %s\n', i, lines{i});
        n_fail = n_fail + 1;
        continue;
    end
    key_path = parts{1}; session_dir = parts{2};
    [key_dir, ~] = fileparts(key_path);
    out_path = fullfile(key_dir, 'training_key_spatial.mat');
    try
        k = load(key_path, 'n_review');
        n_review = double(k.n_review);
        rn = load(fullfile(session_dir, 'review_neuron.mat'), 'neuron');
        neuron = rn.neuron;
        A = full(neuron.A);
        n = size(A, 2);
        spatial_ok = double(n == n_review);
        if ~spatial_ok
            fprintf(2, '  [%d] %s: review_neuron.mat has %d columns, key expects %d -> spatial_ok=0\n', ...
                i, session_dir, n, n_review);
            overlap_max = zeros(n, 1); overlap_partner = zeros(n, 1);
            centroid_yx = zeros(n, 2); n_pixels = zeros(n, 1);
        else
            nr = sqrt(sum(A.^2, 1)) + 1e-12;
            An = bsxfun(@rdivide, A, nr);
            C = An' * An;                      % n x n cosine overlap
            C(1:n+1:end) = -Inf;               % exclude self
            if n > 1
                [overlap_max, overlap_partner] = max(C, [], 2);
                overlap_max = max(overlap_max, 0);
            else
                overlap_max = 0; overlap_partner = 0;
            end
            overlap_max = double(overlap_max(:));
            overlap_partner = double(overlap_partner(:));
            centroid_yx = com(neuron.A, neuron.options.d1, neuron.options.d2);
            centroid_yx = double(centroid_yx(:, 1:2));
            n_pixels = double(sum(A > 0, 1))';
        end
        augmented_at = datestr(now, 'yyyy-mm-dd HH:MM:SS'); %#ok<TNOW1,DATST>
        save(out_path, 'overlap_max', 'overlap_partner', 'centroid_yx', ...
            'n_pixels', 'spatial_ok', 'augmented_at', '-v7');
        fprintf('  [%d] %s: n=%d spatial_ok=%d -> %s\n', i, session_dir, n, spatial_ok, out_path);
        n_ok = n_ok + 1;
    catch err
        fprintf(2, '  [%d] FAILED %s: %s\n', i, session_dir, err.message);
        n_fail = n_fail + 1;
    end
end
fprintf('acorn_training_key_augment done: %d ok, %d failed\n', n_ok, n_fail);
end
