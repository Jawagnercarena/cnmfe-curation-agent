cd('C:/code/CNMF_E_LEGACY_BIANE_CLAUDE');
run('cnmfe_setup.m');
root = 'D:/Julian_CNMFe/vCA1/6odorDualDiffRew';
out = 'C:/Users/julia/AppData/Local/Temp/claude/c--code-CNMF-E-LEGACY-BIANE-CLAUDE/b4949d41-93ee-415a-9a29-2d42de916c5d/scratchpad/review_cnmax.csv';
fid = fopen(out, 'w');
fprintf(fid, 'session,k,cnmax,pnrmax,npix_core,d1,d2\n');
items = dir(root);
for i = 1:numel(items)
    nm = items(i).name;
    sd = fullfile(root, nm);
    if ~items(i).isdir || nm(1) == '.'
        continue;
    end
    rn = fullfile(sd, 'review_neuron.mat');
    lb = fullfile(sd, 'labels.mat');
    cf = fullfile(sd, 'Cn.mat');
    pf = fullfile(sd, 'pnr.mat');
    if ~(exist(rn, 'file') && exist(lb, 'file') && exist(cf, 'file') && exist(pf, 'file'))
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
    C = load(cf);
    Cn = double(C.Cn);
    P = load(pf);
    pnr = double(P.pnr);
    if any(size(Cn) ~= [d1 d2])
        Cn = imresize(Cn, [d1 d2], 'nearest');
    end
    if any(size(pnr) ~= [d1 d2])
        pnr = imresize(pnr, [d1 d2], 'nearest');
    end
    Cn(isnan(Cn)) = 0;
    pnr(isnan(pnr)) = 0;
    N = size(A, 2);
    for k = 1:N
        a = reshape(A(:, k), d1, d2);
        core = a > 0.5 * max(a(:));
        fprintf(fid, '%s,%d,%.5f,%.4f,%d,%d,%d\n', nm, k, max(Cn(core)), max(pnr(core)), nnz(core), d1, d2);
    end
    fprintf('%s: %d review candidates exported\n', nm, N);
end
fclose(fid);
fprintf('EXPORT DONE\n');
