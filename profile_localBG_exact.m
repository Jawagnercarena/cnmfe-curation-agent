%% profile_localBG_exact.m
% Exhaustive equivalence test for a faster local_background inner loop.
%
% The variant race showed the loop is starved by strided gathers: Yb(ind_nhood,:)
% picks ~168 rows out of 65,536 from a column-major array, so every element sits
% on its own cache line.  Fixing that gave 3.90x (time-major) and 14.35x
% (+parfor), but both changed results at ~5e-14 relative, because Xt'*Xt is a
% different BLAS call than X*X'.  "Small round-off" cannot be proven safe
% downstream: merges and trims are threshold comparisons.
%
% So this tests a third option built to be BIT-IDENTICAL:
%
%   Bx  gather contiguously from a transposed copy, then transpose the small
%       (J x nT, ~17 MB) block back, so X holds the same values in the same
%       layout as today and X*X' is literally the same call on the same bytes.
%       Likewise w'*Yb(ind_nhood,:) becomes w'*(Yt(:,ind_nhood).').
%
% If Bx is bit-identical over every pixel, then no candidate, feature, merge or
% trim decision can change, and no end-to-end A/B run is needed to prove it.
%
% Both outputs are compared: the reconstruction AND the weights cell that becomes
% Ybg_weights.mat and is replayed by CNMFe_final_save.m.  The variant race never
% checked the weights at all.
%
% Coverage: exhaustive (all 65,536 px) at the session's own ring radius, plus a
% stratified boundary/interior sample at the extremes of the cohort's geometry
% (18 animal gSig/gSiz combinations, gSiz 22..50 -> rr 18..39).
%
%   session_dir = 'D:\Julian_CNMFe\BLA\3odor\AVG5x-TSeries-07082026-bla36-810um-25z-000';
%   also_test_B = true;
%   rr_sweep    = [18 39];
%   run('C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\profile_localBG_exact.m')

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
cvx_dirs = strsplit(genpath(fullfile(repo_root, 'cvx')), pathsep);
cvx_dirs = cvx_dirs(~cellfun(@isempty, cvx_dirs));
addpath(strjoin(cvx_dirs(~endsWith(cvx_dirs, [filesep 'narginchk_'])), pathsep));
clear cvx_dirs;
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('profile_localBG_exact: session_dir must be set.');
end
if ~exist('also_test_B', 'var') || isempty(also_test_B); also_test_B = true;  end
if ~exist('rr_sweep', 'var');                            rr_sweep = [18 39]; end
if ~exist('n_edge_samp', 'var');  n_edge_samp = 1200; end
if ~exist('n_int_samp', 'var');   n_int_samp  = 800;  end

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');
warning('off', 'MATLAB:singularMatrix');

%% ---- rebuild localBG state up to the loop ----
fprintf('[EXACT] Loading session...\n');
ndata  = load(fullfile(session_dir, 'neuron.mat'));
neuron = ndata.neuron; clear ndata;

[~, session_nm] = fileparts(session_dir);
nam_mat = fullfile(session_dir, [session_nm, '.mat']);
if ~isfile(nam_mat)
    cand   = dir(fullfile(session_dir, '*.mat'));
    reject = {'neuron','Cn','Coor','pnr','Ybg','Ybg_mean', ...
              'spatial_footprints','review_neuron','Ybg_weights'};
    for i = numel(cand):-1:1
        [~, nm] = fileparts(cand(i).name);
        if any(strcmp(nm, reject)); cand(i) = []; end
    end
    nam_mat = fullfile(session_dir, cand(1).name);
end
data_file = matfile(nam_mat);
Ysiz = data_file.Ysiz;
d1 = Ysiz(1); d2 = Ysiz(2); T = Ysiz(3);
Yb = reshape(double(data_file.Y), d1*d2, T) - neuron.A*neuron.C;

ssub   = 2;
rr0    = ceil(neuron.options.gSiz * 1.5);
thresh = 10;
sn     = neuron.reshape(neuron.P.sn, 2);

Yb    = reshape(Yb, d1, d2, T);
Ymean = mean(Yb, 3);
Yb    = Yb - bsxfun(@minus, Ymean, ones(1, 1, T));
Yb    = imresize(Yb, 1./ssub);
[d1s, d2s, ~] = size(Yb);
sn_s  = imresize(sn, 1./ssub);
Npix  = d1s*d2s;
clear Ymean;

rr_production = round(rr0/ssub) + 1;
fprintf('[EXACT] %d x %d x %d; grid %d x %d (%d px); production rr = %d\n', ...
    d1, d2, T, d1s, d2s, Npix, rr_production);

rep = {};
rep{end+1} = sprintf('localBG bit-exactness test — %s', session_nm);
rep{end+1} = datestr(now, 'yyyy-mm-dd HH:MM:SS');
rep{end+1} = sprintf('%d x %d x %d; grid %d x %d (%d px); gSiz %d -> production rr %d', ...
    d1, d2, T, d1s, d2s, Npix, neuron.options.gSiz, rr_production);
rep{end+1} = '';

all_rr = [rr_production, rr_sweep(:)'];

for irr = 1:numel(all_rr)
    rr = all_rr(irr);
    exhaustive = (irr == 1);
    if exhaustive; cov_lbl = 'EXHAUSTIVE (all pixels)'; else; cov_lbl = 'boundary + interior sample'; end
    fprintf('\n[EXACT] ===== rr = %d  (%s) =====\n', rr, cov_lbl);

    rsub_k = (-rr):rr;
    [cind, rind] = meshgrid(rsub_k, rsub_k);
    R = sqrt(cind.^2 + rind.^2);
    neigh_kernel = (R >= rr) .* (R < rr+1);
    J = nnz(neigh_kernel);

    Yconv = bsxfun(@times, imfilter(Yb, neigh_kernel), ...
        1./imfilter(ones(d1s, d2s), neigh_kernel));
    ind_event = (bsxfun(@times, Yb - Yconv, 1./sn_s) > thresh);
    Ywork = Yb;
    Ywork(ind_event) = Yconv(ind_event);
    ind_event = reshape(ind_event, Npix, []);
    clear Yconv;

    [r_shift, c_shift] = find(neigh_kernel);
    r_shift = r_shift - rr - 1;
    c_shift = c_shift - rr - 1;
    [csub, rsub] = meshgrid(1:d2s, 1:d1s);
    csub = reshape(csub, [], 1); rsub = reshape(rsub, [], 1);
    csub = bsxfun(@plus, csub, c_shift'); rsub = bsxfun(@plus, rsub, r_shift');
    oob = or(or(csub<1, csub>d2s), or(rsub<1, rsub>d1s));
    csub(oob) = nan; rsub(oob) = nan;

    Ywork = reshape(Ywork, Npix, []);

    % Pixel selection.  Exhaustive at the production geometry.  At the sweep
    % radii the clipped set alone is 17k-34k pixels, so sample it — the boundary
    % path differs from the interior only in how many neighbours survive, and a
    % stratified sample exercises every such length.
    n_clip = sum(isnan(rsub) | isnan(csub), 2);
    if exhaustive
        test_px = 1:Npix;
    else
        rng(0);
        edge_px = find(n_clip > 0);
        int_px  = find(n_clip == 0);
        e = edge_px(randperm(numel(edge_px), min(n_edge_samp, numel(edge_px))));
        if isempty(int_px)
            s = [];
        else
            s = int_px(randperm(numel(int_px), min(n_int_samp, numel(int_px))));
        end
        test_px = sort([e(:); s(:)])';
    end
    nt = numel(test_px);
    fprintf('[EXACT] J=%d; testing %d px (%d boundary-clipped, %d with no valid neighbour)\n', ...
        J, nt, sum(n_clip(test_px) > 0), sum(n_clip(test_px) == J));

    % ---- A: current implementation (reference) ----
    YestA = zeros(nt, T);  wA = cell(nt, 1);
    tA = tic;
    for k = 1:nt
        px    = test_px(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood); continue; end
        tmp_ind = ~ind_event(px, 2:end);
        X     = Ywork(ind_nhood, tmp_ind);
        y     = Ywork(px, tmp_ind);
        tmpXX = X*X';
        tmpXy = X*y';
        w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
        YestA(k,:) = w'*Ywork(ind_nhood,:);
        wA{k} = [ind_nhood; w'];
    end
    sA = toc(tA);
    fprintf('[EXACT] A  current               %8.1f s\n', sA);

    tt = tic; Yt = Ywork.'; s_tp = toc(tt);

    v_name = {}; v_time = {}; v_Y = {}; v_w = {};

    % ---- Bx: contiguous gather, transposed back — every gemm identical ----
    YestX = zeros(nt, T);  wX = cell(nt, 1);
    tX = tic;
    for k = 1:nt
        px    = test_px(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood); continue; end
        tmp_ind = ~ind_event(px, 2:end);
        X     = Yt(tmp_ind, ind_nhood).';
        y     = Yt(tmp_ind, px).';
        tmpXX = X*X';
        tmpXy = X*y';
        w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
        YestX(k,:) = w'*(Yt(:, ind_nhood).');
        wX{k} = [ind_nhood; w'];
    end
    sX = toc(tX);
    v_name{end+1} = 'Bx contiguous+transpose'; v_time{end+1} = sX;
    v_Y{end+1} = YestX; v_w{end+1} = wX;

    % ---- B: pure time-major (expected ~5e-14, kept for comparison) ----
    if also_test_B
        YestB = zeros(nt, T); wB = cell(nt, 1);
        tB = tic;
        for k = 1:nt
            px    = test_px(k);
            valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
            ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
            if isempty(ind_nhood); continue; end
            tmp_ind = ~ind_event(px, 2:end);
            Xt    = Yt(tmp_ind, ind_nhood);
            yt    = Yt(tmp_ind, px);
            tmpXX = Xt'*Xt;
            tmpXy = Xt'*yt;
            w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
            YestB(k,:) = (Yt(:, ind_nhood)*w).';
            wB{k} = [ind_nhood; w'];
        end
        sB = toc(tB);
        v_name{end+1} = 'B  pure time-major'; v_time{end+1} = sB;
        v_Y{end+1} = YestB; v_w{end+1} = wB;
    end

    rep{end+1} = sprintf('rr = %d   J = %d neighbours   %s   %d px tested', ...
        rr, J, cov_lbl, nt);
    rep{end+1} = sprintf('  A  current               %8.1f s     (reference)', sA);

    for v = 1:numel(v_name)
        YB = v_Y{v}; wB2 = v_w{v};
        dY = max(abs(YestA(:) - YB(:)));
        nbad = 0; dW = 0;
        for k = 1:nt
            a = wA{k}; b = wB2{k};
            if isempty(a) && isempty(b); continue; end
            if (isempty(a) ~= isempty(b)) || ~isequal(size(a), size(b)) ...
                    || ~isequal(a(1,:), b(1,:))
                nbad = nbad + 1;
                continue;
            end
            dW = max(dW, max(abs(a(2,:) - b(2,:))));
        end
        if dY == 0 && nbad == 0 && dW == 0
            vd = '   <== BIT-IDENTICAL';
        else
            vd = '';
        end
        fprintf('[EXACT] %-24s %8.1f s  %5.2fx | Yest %.3g | idx bad %d | w %.3g%s\n', ...
            v_name{v}, v_time{v}, sA/v_time{v}, dY, nbad, dW, vd);
        rep{end+1} = sprintf('  %-24s %8.1f s  %5.2fx  (transpose %.1f s)', ...
            v_name{v}, v_time{v}, sA/v_time{v}, s_tp);
        rep{end+1} = sprintf('     Yest max|diff| %-10.3g  weights idx mismatches %-4d  weights max|diff| %-10.3g%s', ...
            dY, nbad, dW, vd);
    end
    rep{end+1} = '';

    clear Yt YestA YestX wA wX v_Y v_w Ywork ind_event rsub csub;
    if also_test_B; clear YestB wB; end
end

out = fullfile(session_dir, 'localBG_exact.txt');
txt = strjoin(rep, newline);
fprintf('\n%s\n', txt);
fid = fopen(out, 'w'); fprintf(fid, '%s\n', txt); fclose(fid);
fprintf('[EXACT] Report written to %s\n', out);
