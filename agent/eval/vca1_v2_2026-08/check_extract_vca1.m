% check_extract_vca1.m -- bit-identity of the .feature_expansion extraction
% against review_neuron.mat for every vCA1 labeled agent session.
%
% Port of agent/eval/redteam_2026-08/c1_matlab_full79.m (which returned 79/79
% max|diff| = 0 for BLA).  This is the check Python cannot do: review_neuron.mat
% holds an opaque MCOS Sources2D object, so only MATLAB can reopen it and
% compare the arrays the extraction claims to have copied.
%
% Load-only; writes nothing.
repo_root = 'c:/code/CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
listfile = fullfile(repo_root, 'agent/eval/vca1_v2_2026-08/vca1_extract_sessions.txt');
base = 'D:/Julian_CNMFe/vCA1';
ext = fullfile(base, '.feature_expansion');

fid = fopen(listfile, 'r');
if fid < 0; error('cannot open session list: %s', listfile); end
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
            fprintf(2, 'FAIL %s: C size %dx%d vs %dx%d\n', rel, ...
                size(C_rn,1), size(C_rn,2), size(ex.C_raw,1), size(ex.C_raw,2));
            n_fail = n_fail + 1; continue;
        end
        if ~isequal(size(A_rn), size(ex.A))
            fprintf(2, 'FAIL %s: A size %dx%d vs %dx%d\n', rel, ...
                size(A_rn,1), size(A_rn,2), size(ex.A,1), size(ex.A,2));
            n_fail = n_fail + 1; continue;
        end
        dC = full(max(abs(C_rn(:) - ex.C_raw(:))));
        dA = full(max(abs(A_rn(:) - ex.A(:))));
        if dC == 0 && dA == 0
            n_pass = n_pass + 1;
            fprintf('PASS %s (N=%d T=%d)\n', rel, size(C_rn,1), size(C_rn,2));
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
