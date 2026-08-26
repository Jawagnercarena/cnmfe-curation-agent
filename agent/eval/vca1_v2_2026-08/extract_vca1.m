% extract_vca1.m -- vCA1 candidate-data extraction (Step 3a).
%
% Port of agent/eval/step2_2026-08/extract_step2.m for vCA1.  For every labeled
% agent session in vca1_extract_sessions.txt, pull the REVIEW-SET candidate data
% out of review_neuron.mat (C_raw full, A sparse, Cn, dims) into one dot-prefixed
% analysis dir on local D: (dot prefix = invisible to every scanner/watcher).
%
% Needed because CNMFe_final_save.m overwrites C_raw.txt / spatial_footprints.mat
% with the FINAL neuron set at review completion, so the candidate-level traces
% that the v2b features are computed from survive only inside review_neuron.mat.
%
% Resumable: skips sessions whose output already exists.  Never writes into
% session dirs; never modifies anything existing.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
listfile = fullfile(repo_root, 'agent/eval/vca1_v2_2026-08/vca1_extract_sessions.txt');
base = 'D:/Julian_CNMFe/vCA1';
outdir = fullfile(base, '.feature_expansion');
if ~exist(outdir, 'dir'); mkdir(outdir); end

fid = fopen(listfile, 'r');
if fid < 0; error('cannot open session list: %s', listfile); end
sessions = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    if ~isempty(strtrim(ln)); sessions{end+1} = strtrim(ln); end %#ok<AGROW>
end
fclose(fid);
fprintf('extracting %d sessions -> %s\n', numel(sessions), outdir);

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
