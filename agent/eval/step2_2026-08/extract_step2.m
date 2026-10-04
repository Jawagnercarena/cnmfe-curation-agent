% extract_step2.m -- Step 2 candidate-data extraction.
% For every labeled agent session, pull the REVIEW-SET candidate data out of
% review_neuron.mat (C_raw full, A sparse, Cn, dims) into one dot-prefixed
% analysis dir on local D: (dot prefix = invisible to every scanner/watcher).
% Resumable: skips sessions whose output already exists. Never writes into
% session dirs; never modifies anything existing.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
sp = 'C:/Users/julia/AppData/Local/Temp/claude/c--code-CNMF-E-LEGACY-BIANE-CLAUDE/6d3eba36-4249-44c3-b84d-fcd0b93d2660/scratchpad';
base = 'D:/Julian_CNMFe/BLA';
outdir = fullfile(base, '.feature_expansion');
if ~exist(outdir, 'dir'); mkdir(outdir); end

fid = fopen(fullfile(sp, 'step2_sessions.txt'), 'r');
sessions = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    if ~isempty(strtrim(ln)); sessions{end+1} = strtrim(ln); end %#ok<AGROW>
end
fclose(fid);
fprintf('extracting %d sessions\n', numel(sessions));

n_ok = 0; n_skip = 0; n_fail = 0;
for i = 1:numel(sessions)
    rel = sessions{i};
    outname = fullfile(outdir, [strrep(rel, '/', '__') '.mat']);
    if exist(outname, 'file'); n_skip = n_skip + 1; continue; end
    try
        rn = load(fullfile(base, rel, 'review_neuron.mat'));
        C_raw = full(rn.neuron.C_raw);          % N x T
        A = sparse(rn.neuron.A);                % pixels x N, keep sparse
        if isfield(rn, 'Cn'); Cn = rn.Cn; else; cnl = load(fullfile(base, rel, 'Cn.mat')); Cn = cnl.Cn; end
        d1 = size(Cn, 1); d2 = size(Cn, 2);
        if size(A, 1) ~= d1 * d2
            error('A rows %d != d1*d2 %d', size(A, 1), d1 * d2);
        end
        save(outname, 'C_raw', 'A', 'Cn', 'd1', 'd2', '-v7');
        n_ok = n_ok + 1;
        fprintf('OK %s: N=%d T=%d\n', rel, size(C_raw, 1), size(C_raw, 2));
    catch err
        n_fail = n_fail + 1;
        fprintf(2, 'FAILED %s: %s\n', rel, err.message);
    end
end
fprintf('done: %d ok, %d skipped, %d failed\n', n_ok, n_skip, n_fail);
