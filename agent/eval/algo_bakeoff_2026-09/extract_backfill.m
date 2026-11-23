function extract_backfill(area, list_file)
% extract_backfill -- pull the REVIEW-SET candidate data (C_raw full, A sparse,
% Cn, dims) out of review_neuron.mat for the labeled agent sessions that have
% no .feature_expansion extraction yet (the 6 BLA / 8 vCA1 / 17 DG_AL sessions
% listed by prep_raw_corpus' inventory, results/backfill_<AREA>_sessions.txt).
%
% Same output format and location as agent/eval/step2_2026-08/extract_step2.m
% (<DATA_PARENT>/<AREA>/.feature_expansion/<task>__<session>.mat, -v7), so
% rt_lib.load_extraction and prep_raw_corpus.py read them unchanged.
% Resumable: skips sessions whose output exists.  Never writes into a session
% dir; never modifies anything existing.
%
% Usage (batch):  matlab -batch "cd('<this dir>'); extract_backfill('DG_AL', 'results/backfill_DG_AL_sessions.txt')"
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
data_parent = getenv('CNMFE_DATA_PARENT');
if isempty(data_parent); data_parent = 'D:/Julian_CNMFe'; end
base = fullfile(data_parent, area);
outdir = fullfile(base, '.feature_expansion');
if ~exist(outdir, 'dir'); mkdir(outdir); end

fid = fopen(list_file, 'r');
if fid < 0; error('cannot open %s', list_file); end
sessions = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    if ~isempty(strtrim(ln)); sessions{end+1} = strtrim(ln); end %#ok<AGROW>
end
fclose(fid);
fprintf('[%s] extracting %d sessions into %s\n', area, numel(sessions), outdir);

n_ok = 0; n_skip = 0; n_fail = 0;
for i = 1:numel(sessions)
    rel = sessions{i};
    outname = fullfile(outdir, [strrep(rel, '/', '__') '.mat']);
    if exist(outname, 'file'); n_skip = n_skip + 1; continue; end
    try
        t0 = tic;
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
        fprintf('OK %s: N=%d T=%d (%.0f s)\n', rel, size(C_raw, 1), size(C_raw, 2), toc(t0));
    catch err
        n_fail = n_fail + 1;
        fprintf(2, 'FAILED %s: %s\n', rel, err.message);
    end
end
fprintf('[%s] done: %d ok, %d skipped, %d failed\n', area, n_ok, n_skip, n_fail);
end
