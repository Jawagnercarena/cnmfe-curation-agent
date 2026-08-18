% C1 extension: bit-identity of .feature_expansion extraction vs
% review_neuron.mat for ALL 79 sessions. Load-only; writes nothing.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
base = 'D:/Julian_CNMFe/BLA';
ext = fullfile(base, '.feature_expansion');
fid = fopen(fullfile(ext, '_pinned', 'step2_sessions.txt'), 'r');
sessions = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    if ~isempty(strtrim(ln)); sessions{end+1} = strtrim(ln); end %#ok<AGROW>
end
fclose(fid);
n_pass = 0; n_fail = 0;
for i = 1:numel(sessions)
    rel = sessions{i};
    try
        rn = load(fullfile(base, rel, 'review_neuron.mat'));
        ex = load(fullfile(ext, [strrep(rel, '/', '__') '.mat']));
        C_rn = full(rn.neuron.C_raw);
        A_rn = sparse(rn.neuron.A);
        if ~isequal(size(C_rn), size(ex.C_raw))
            fprintf(2, 'FAIL %s: size %dx%d vs %dx%d\n', rel, ...
                size(C_rn,1), size(C_rn,2), size(ex.C_raw,1), size(ex.C_raw,2));
            n_fail = n_fail + 1; continue;
        end
        dC = full(max(abs(C_rn(:) - ex.C_raw(:))));
        dA = full(max(abs(A_rn(:) - ex.A(:))));
        if dC == 0 && dA == 0
            n_pass = n_pass + 1;
            fprintf('PASS %s (N=%d)\n', rel, size(C_rn,1));
        else
            n_fail = n_fail + 1;
            fprintf(2, 'FAIL %s: maxdiff C=%.3g A=%.3g\n', rel, dC, dA);
        end
    catch err
        n_fail = n_fail + 1;
        fprintf(2, 'FAIL %s: %s\n', rel, err.message);
    end
end
fprintf('SUMMARY: %d pass, %d fail of %d\n', n_pass, n_fail, numel(sessions));
