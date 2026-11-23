cd('C:/code/CNMF_E_LEGACY_BIANE_CLAUDE');
run('cnmfe_setup.m');
root = 'D:/Julian_CNMFe/vCA1/6odorDualDiffRew';
out = 'C:/Users/julia/AppData/Local/Temp/claude/c--code-CNMF-E-LEGACY-BIANE-CLAUDE/b4949d41-93ee-415a-9a29-2d42de916c5d/scratchpad/review_peaks.csv';
fid = fopen(out, 'w');
fprintf(fid, 'session,k,peak_row,peak_col,area_px,d1,d2\n');
items = dir(root);
for i = 1:numel(items)
    nm = items(i).name;
    sd = fullfile(root, nm);
    if ~items(i).isdir || nm(1) == '.'
        continue;
    end
    rn = fullfile(sd, 'review_neuron.mat');
    lb = fullfile(sd, 'labels.mat');
    if ~(exist(rn, 'file') && exist(lb, 'file'))
        continue;
    end
    S = load(rn);
    neuron = S.neuron;
    A = full(neuron.A);
    d1 = neuron.options.d1;
    d2 = neuron.options.d2;
    if size(A, 1) ~= d1 * d2
        fprintf('SKIP %s: A has %d rows, d1*d2 = %d\n', nm, size(A, 1), d1 * d2);
        continue;
    end
    N = size(A, 2);
    for k = 1:N
        a = reshape(A(:, k), d1, d2);
        [~, idx] = max(a(:));
        [r, c] = ind2sub([d1 d2], idx);
        fprintf(fid, '%s,%d,%d,%d,%d,%d,%d\n', nm, k, r, c, nnz(a > 0.5 * max(a(:))), d1, d2);
    end
    fprintf('%s: %d peaks exported\n', nm, N);
end
fclose(fid);
fprintf('EXPORT DONE\n');
