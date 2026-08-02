%% profile_localBG_confirm.m
% Final confirmation run before local_background.m is changed.
%
% Races the two surviving candidates against the current implementation and
% verifies both outputs (the reconstruction AND the weights cell that becomes
% Ybg_weights.mat) over EVERY pixel at the production geometry, plus large
% boundary-weighted samples at the extremes of the cohort's ring radius.
%
%   A     current code                                            (reference)
%   Bx2a  contiguous gather with the transpose MATERIALISED into a variable, so
%         every BLAS call is the same call on the same bytes -> expected exact
%   B     pure time-major (Xt'*Xt), a different BLAS call -> expected ~5e-12
%
% Earlier runs established: A = 2424.9 s exhaustively (production logs 2605.93 s);
% Bx2a bit-identical on 6000 px at rr=28; B 4.04x with max|diff| 5.14e-12.  This
% run is the exhaustive check that was still owed for Bx2a's reconstruction line.
%
%   session_dir = 'D:\Julian_CNMFe\BLA\3odor\AVG5x-TSeries-07082026-bla36-810um-25z-000';
%   rr_sweep    = [18 39];
%   n_sweep     = 8000;
%   run('C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\profile_localBG_confirm.m')

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
cvx_dirs = strsplit(genpath(fullfile(repo_root, 'cvx')), pathsep);
cvx_dirs = cvx_dirs(~cellfun(@isempty, cvx_dirs));
addpath(strjoin(cvx_dirs(~endsWith(cvx_dirs, [filesep 'narginchk_'])), pathsep));
clear cvx_dirs;
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('profile_localBG_confirm: session_dir must be set.');
end
if ~exist('rr_sweep', 'var');                    rr_sweep = [18 39]; end
if ~exist('n_sweep', 'var') || isempty(n_sweep); n_sweep  = 8000;    end

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');
warning('off', 'MATLAB:singularMatrix');

%% ---- rebuild localBG state ----
fprintf('[CONFIRM] Loading session...\n');
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
Ysiz = data_file.Ysiz; d1 = Ysiz(1); d2 = Ysiz(2); T = Ysiz(3);
Yb = reshape(double(data_file.Y), d1*d2, T) - neuron.A*neuron.C;

ssub = 2; thresh = 10;
rr0  = ceil(neuron.options.gSiz * 1.5);
sn   = neuron.reshape(neuron.P.sn, 2);
Yb   = reshape(Yb, d1, d2, T);
Ymean = mean(Yb, 3);
Yb   = Yb - bsxfun(@minus, Ymean, ones(1, 1, T));
Yb   = imresize(Yb, 1./ssub);
[d1s, d2s, ~] = size(Yb);
sn_s = imresize(sn, 1./ssub);
Npix = d1s*d2s;
clear Ymean;
rr_production = round(rr0/ssub) + 1;

fprintf('[CONFIRM] %d x %d x %d; grid %d x %d (%d px); production rr = %d\n', ...
    d1, d2, T, d1s, d2s, Npix, rr_production);

rep = {};
rep{end+1} = sprintf('localBG final confirmation — %s', session_nm);
rep{end+1} = datestr(now, 'yyyy-mm-dd HH:MM:SS');
rep{end+1} = sprintf('%d x %d x %d; grid %d x %d (%d px); gSiz %d -> production rr %d', ...
    d1, d2, T, d1s, d2s, Npix, neuron.options.gSiz, rr_production);
rep{end+1} = '';

all_rr = [rr_production, rr_sweep(:)'];

for irr = 1:numel(all_rr)
    rr = all_rr(irr);
    exhaustive = (irr == 1);
    if exhaustive
        cov_lbl = 'EXHAUSTIVE (every pixel)';
    else
        cov_lbl = sprintf('boundary-weighted sample (%d px)', n_sweep);
    end
    fprintf('\n[CONFIRM] ===== rr = %d  (%s) =====\n', rr, cov_lbl);

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
    r_shift = r_shift - rr - 1; c_shift = c_shift - rr - 1;
    [csub, rsub] = meshgrid(1:d2s, 1:d1s);
    csub = reshape(csub, [], 1); rsub = reshape(rsub, [], 1);
    csub = bsxfun(@plus, csub, c_shift'); rsub = bsxfun(@plus, rsub, r_shift');
    oob = or(or(csub<1, csub>d2s), or(rsub<1, rsub>d1s));
    csub(oob) = nan; rsub(oob) = nan;
    Ywork = reshape(Ywork, Npix, []);

    n_clip = sum(isnan(rsub) | isnan(csub), 2);
    if exhaustive
        test_px = 1:Npix;
    else
        rng(0);
        edge_px = find(n_clip > 0); int_px = find(n_clip == 0);
        ne = min(round(n_sweep*0.625), numel(edge_px));
        ni = min(n_sweep - ne, numel(int_px));
        test_px = sort([edge_px(randperm(numel(edge_px), ne)); ...
                        int_px(randperm(numel(int_px), ni))])';
    end
    nt = numel(test_px);
    fprintf('[CONFIRM] J=%d; %d px tested (%d boundary-clipped)\n', ...
        J, nt, sum(n_clip(test_px) > 0));

    % ---- A: reference ----
    YestA = zeros(nt, T); wA = cell(nt, 1);
    tA = tic;
    for k = 1:nt
        px = test_px(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood); continue; end
        tmp_ind = ~ind_event(px, 2:end);
        X = Ywork(ind_nhood, tmp_ind);
        y = Ywork(px, tmp_ind);
        tmpXX = X*X'; tmpXy = X*y';
        w = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
        YestA(k,:) = w'*Ywork(ind_nhood,:);
        wA{k} = [ind_nhood; w'];
    end
    sA = toc(tA);
    fprintf('[CONFIRM] A     current               %8.1f s\n', sA);

    tt = tic; Yt = Ywork.'; s_tp = toc(tt);
    names = {}; times = {}; Ys = {}; ws = {};

    % ---- Bx2a: contiguous gather, transposes materialised ----
    Yest1 = zeros(nt, T); w1 = cell(nt, 1);
    t1 = tic;
    for k = 1:nt
        px = test_px(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood); continue; end
        tmp_ind = ~ind_event(px, 2:end);
        X = Yt(tmp_ind, ind_nhood).';     % materialised
        y = Yt(tmp_ind, px).';
        tmpXX = X*X'; tmpXy = X*y';
        w = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
        M = Yt(:, ind_nhood).';           % materialised
        Yest1(k,:) = w'*M;
        w1{k} = [ind_nhood; w'];
    end
    names{end+1} = 'Bx2a materialised'; times{end+1} = toc(t1);
    Ys{end+1} = Yest1; ws{end+1} = w1;

    % ---- B: pure time-major ----
    Yest2 = zeros(nt, T); w2 = cell(nt, 1);
    t2 = tic;
    for k = 1:nt
        px = test_px(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood); continue; end
        tmp_ind = ~ind_event(px, 2:end);
        Xt = Yt(tmp_ind, ind_nhood);
        yt = Yt(tmp_ind, px);
        tmpXX = Xt'*Xt; tmpXy = Xt'*yt;
        w = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
        Yest2(k,:) = (Yt(:, ind_nhood)*w).';
        w2{k} = [ind_nhood; w'];
    end
    names{end+1} = 'B    time-major'; times{end+1} = toc(t2);
    Ys{end+1} = Yest2; ws{end+1} = w2;

    rep{end+1} = sprintf('rr = %d   J = %d neighbours   %s   %d px', ...
        rr, J, cov_lbl, nt);
    rep{end+1} = sprintf('  A     current             %8.1f s            (reference)', sA);

    for v = 1:numel(names)
        YB = Ys{v}; wB = ws{v};
        dabs = abs(YestA(:) - YB(:));
        dY   = max(dabs);
        dmed = median(dabs);
        nnzd = nnz(dabs);
        nbad = 0; dW = 0;
        for k = 1:nt
            a = wA{k}; b = wB{k};
            if isempty(a) && isempty(b); continue; end
            if (isempty(a) ~= isempty(b)) || ~isequal(size(a), size(b)) ...
                    || ~isequal(a(1,:), b(1,:))
                nbad = nbad + 1; continue;
            end
            dW = max(dW, max(abs(a(2,:) - b(2,:))));
        end
        if dY == 0 && nbad == 0 && dW == 0
            vd = '  <== BIT-IDENTICAL';
        else
            vd = '';
        end
        fprintf('[CONFIRM] %-21s %8.1f s %5.2fx | Yest max %.3g (%d of %d elems differ) | idx bad %d | w %.3g%s\n', ...
            names{v}, times{v}, sA/times{v}, dY, nnzd, numel(dabs), nbad, dW, vd);
        rep{end+1} = sprintf('  %-21s %8.1f s  %5.2fx', names{v}, times{v}, sA/times{v});
        rep{end+1} = sprintf('     Yest max|diff| %-10.3g median %-10.3g  differing elements %d of %d', ...
            dY, dmed, nnzd, numel(dabs));
        rep{end+1} = sprintf('     weights idx mismatches %-4d  weights max|diff| %-10.3g%s', ...
            nbad, dW, vd);
    end
    rep{end+1} = sprintf('  (one-off transpose %.1f s)', s_tp);
    rep{end+1} = '';

    clear Yt YestA Yest1 Yest2 wA w1 w2 Ys ws Ywork ind_event rsub csub;
end

txt = strjoin(rep, newline);
fprintf('\n%s\n', txt);
out = fullfile(session_dir, 'localBG_confirm.txt');
fid = fopen(out, 'w'); fprintf(fid, '%s\n', txt); fclose(fid);
fprintf('[CONFIRM] Report written to %s\n', out);
