%% profile_localBG_variants.m
% Follow-up to profile_localBG.m, which established that the per-pixel ring
% regression is 99.3% of the full-resolution background estimate and that plain
% parfor buys only 2.29x (and degrades above 6 workers).
%
% This script races candidate rewrites against the current implementation on one
% identical pixel probe, and reports how far each one's output drifts from the
% current arithmetic.  Read-only: writes localBG_variants.txt into the session.
%
%   session_dir   = 'D:\Julian_CNMFe\BLA\3odor\AVG5x-TSeries-07082026-bla36-810um-25z-000';
%   n_probe       = 6000;            % big enough that lazy per-worker transfer amortises
%   worker_counts = [4 6 12 24];
%   run('C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\profile_localBG_variants.m')
%
% Variants:
%   A  baseline        current code: column-major Yb, serial, 24-thread client BLAS
%   B  time-major      Yb transposed to T x Npix so each neighbour's samples are
%                      contiguous -- targets the ~8x cache-line amplification that
%                      makes A gather-bound (~18 GFLOP/s measured)
%   C  parfor          A inside parfor, Constant pre-warmed so the 6.7 GB per-worker
%                      transfer is NOT inside the timed region (the confound in the
%                      first profile)
%   D  B + parfor      best of both, at C's best worker count
%   E  B + frame skip  fits w on every 3rd frame, reconstructs on all -- quantifies
%                      the "wildly over-determined" option.  Changes results
%                      materially; measured here, not recommended by this script.

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
cvx_dirs = strsplit(genpath(fullfile(repo_root, 'cvx')), pathsep);
cvx_dirs = cvx_dirs(~cellfun(@isempty, cvx_dirs));
addpath(strjoin(cvx_dirs(~endsWith(cvx_dirs, [filesep 'narginchk_'])), pathsep));
clear cvx_dirs;
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('profile_localBG_variants: session_dir must be set.');
end
if ~exist('n_probe', 'var')       || isempty(n_probe);       n_probe = 6000;          end
if ~exist('worker_counts', 'var') || isempty(worker_counts); worker_counts = [4 6 12 24]; end
if ~exist('frame_skip', 'var')    || isempty(frame_skip);    frame_skip = 3;          end

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');
warning('off', 'MATLAB:singularMatrix');

%% ---- Rebuild localBG's state up to the loop (same as profile_localBG.m) ----
fprintf('[VAR] Loading session...\n');
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
Y  = reshape(double(data_file.Y), d1*d2, T);

Yb = Y - neuron.A*neuron.C;
clear Y;

ssub   = 2;
rr0    = ceil(neuron.options.gSiz * 1.5);
thresh = 10;
sn     = neuron.reshape(neuron.P.sn, 2);

Yb    = reshape(Yb, d1, d2, T);
Ymean = mean(Yb, 3);
Yb    = Yb - bsxfun(@minus, Ymean, ones(1, 1, T));
Yb    = imresize(Yb, 1./ssub);
[d1s, d2s, ~] = size(Yb);
rr    = round(rr0/ssub) + 1;
sn_s  = imresize(sn, 1./ssub);

rsub_k = (-rr):rr;
[cind, rind] = meshgrid(rsub_k, rsub_k);
R = sqrt(cind.^2 + rind.^2);
neigh_kernel = (R >= rr) .* (R < rr+1);
Yconv = bsxfun(@times, imfilter(Yb, neigh_kernel), ...
    1./imfilter(ones(d1s, d2s), neigh_kernel));
ind_event = (bsxfun(@times, Yb - Yconv, 1./sn_s) > thresh);
Yb(ind_event) = Yconv(ind_event);
ind_event = reshape(ind_event, d1s*d2s, []);
clear Yconv;

[r_shift, c_shift] = find(neigh_kernel);
r_shift = r_shift - rr - 1;
c_shift = c_shift - rr - 1;
[csub, rsub] = meshgrid(1:d2s, 1:d1s);
csub = reshape(csub, [], 1); rsub = reshape(rsub, [], 1);
csub = bsxfun(@plus, csub, c_shift'); rsub = bsxfun(@plus, rsub, r_shift');
oob = or(or(csub<1, csub>d2s), or(rsub<1, rsub>d1s));
csub(oob) = nan; rsub(oob) = nan;

Yb   = reshape(Yb, d1s*d2s, []);
Npix = d1s*d2s;

rng(0);
n_probe = min(n_probe, Npix);
probe   = sort(randperm(Npix, n_probe));
fprintf('[VAR] %d x %d x %d; grid %d x %d (%d px); %d neighbours; probe %d px\n', ...
    d1, d2, T, d1s, d2s, Npix, nnz(neigh_kernel), n_probe);
fprintf('[VAR] transient-suppressed samples: %.4f%% (0 => tmp_ind is all-true)\n', ...
    100*nnz(ind_event)/numel(ind_event));

res = struct('name', {}, 'loop_s', {}, 'prep_s', {}, 'speedup', {}, ...
             'max_abs', {}, 'rel', {}, 'note', {});

%% ---- A: baseline (the current code) ----
fprintf('[VAR] A: baseline serial, column-major...\n');
rowsA = zeros(n_probe, T);
tA = tic;
for k = 1:n_probe
    px    = probe(k);
    valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
    ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
    if isempty(ind_nhood); continue; end
    tmp_ind = ~ind_event(px, 2:end);
    X     = Yb(ind_nhood, tmp_ind);
    y     = Yb(px, tmp_ind);
    tmpXX = X*X';
    tmpXy = X*y';
    w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
    rowsA(k,:) = w'*Yb(ind_nhood,:);
end
sA = toc(tA);
scaleA = max(abs(rowsA(:)));
res(end+1) = struct('name','A baseline (serial, column-major)','loop_s',sA, ...
    'prep_s',0,'speedup',1,'max_abs',0,'rel',0,'note','reference');
fprintf('[VAR]   A: %.1f s\n', sA);

%% ---- B: time-major ----
fprintf('[VAR] B: transposing to time-major (%.1f GB)...\n', numel(Yb)*8/2^30);
tp = tic; Yt = Yb.'; prepB = toc(tp);
fprintf('[VAR]   transpose took %.1f s\n', prepB);

rowsB = zeros(n_probe, T);
tB = tic;
for k = 1:n_probe
    px    = probe(k);
    valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
    ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
    if isempty(ind_nhood); continue; end
    tmp_ind = ~ind_event(px, 2:end);
    Xt    = Yt(tmp_ind, ind_nhood);      % contiguous columns
    yt    = Yt(tmp_ind, px);
    tmpXX = Xt'*Xt;
    tmpXy = Xt'*yt;
    w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
    rowsB(k,:) = (Yt(:, ind_nhood)*w).';
end
sB = toc(tB);
mB = max(abs(rowsB(:) - rowsA(:)));
res(end+1) = struct('name','B time-major (serial)','loop_s',sB,'prep_s',prepB, ...
    'speedup',sA/sB,'max_abs',mB,'rel',mB/max(scaleA,eps), ...
    'note',sprintf('one-off transpose %.1f s', prepB));
fprintf('[VAR]   B: %.1f s (%.2fx), max|diff| %.3g\n', sB, sA/sB, mB);

%% ---- C: parfor on the baseline layout, transfer pre-warmed ----
per_worker_gb = (numel(Yb)*8 + numel(ind_event) + numel(rsub)*8*2)/2^30;
% Guard on FREE PHYSICAL memory, not MemAvailableAllArrays -- the latter counts
% the pagefile, and paging a worker's 7 GB copy would silently turn a scaling
% measurement into a disk benchmark.
try
    mm = memory; free_gb = mm.PhysicalMemory.Available/2^30;
catch
    free_gb = NaN;
end
% `memory` returned nothing usable on the first run, which disabled the guard and
% let the 24-worker config attempt ~173 GB of Constant serialisation -- it died
% with "parallel.pool.Constant is invalid" and took variants D and E down with it.
% Fall back to a hard cap rather than to no guard at all.
if isnan(free_gb)
    hard_cap = 12;
    fprintf('[VAR] `memory` unavailable — capping workers at %d.\n', hard_cap);
    worker_counts = worker_counts(worker_counts <= hard_cap);
end
fprintf('[VAR] %.1f GB per worker; %.0f GB free physical\n', per_worker_gb, free_gb);
best_nw = NaN; best_s = inf;
for nw = worker_counts
    if ~isnan(free_gb) && nw*per_worker_gb > 0.50*free_gb
        fprintf('[VAR] C: skipping %d workers (%.0f GB needed, %.0f GB free)\n', ...
            nw, nw*per_worker_gb, free_gb);
        continue;
    end
  try
    delete(gcp('nocreate'));
    parpool('Processes', nw);
    Yc = parallel.pool.Constant(Yb);
    Ec = parallel.pool.Constant(ind_event);

    % Force the lazy per-worker transfer BEFORE timing -- this is the confound
    % that made the first profile's 8/12-worker numbers look like contention.
    tw = tic;
    nwarm = nw*4;
    warm = zeros(nwarm,1);
    parfor k = 1:nwarm
        Yw = Yc.Value; Ew = Ec.Value;
        warm(k) = Yw(1,1) + double(Ew(1,1));
    end
    warm_s = toc(tw);

    rowsC = zeros(n_probe, T);
    tC = tic;
    parfor k = 1:n_probe
        warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:SingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
        px    = probe(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood)
            rowsC(k,:) = 0;
        else
            Yw = Yc.Value; Ew = Ec.Value;
            tmp_ind = ~Ew(px, 2:end);
            X     = Yw(ind_nhood, tmp_ind);
            y     = Yw(px, tmp_ind);
            tmpXX = X*X';
            tmpXy = X*y';
            w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
            rowsC(k,:) = w'*Yw(ind_nhood,:);
        end
    end
    sC = toc(tC);
    mC = max(abs(rowsC(:) - rowsA(:)));
    res(end+1) = struct('name',sprintf('C parfor col-major, %d workers', nw), ...
        'loop_s',sC,'prep_s',warm_s,'speedup',sA/sC,'max_abs',mC, ...
        'rel',mC/max(scaleA,eps), ...
        'note',sprintf('warm-up (transfer) %.1f s', warm_s));  %#ok<SAGROW>
    fprintf('[VAR]   C %2d workers: %.1f s (%.2fx), warm %.1f s, max|diff| %.3g\n', ...
        nw, sC, sA/sC, warm_s, mC);
    if sC < best_s; best_s = sC; best_nw = nw; end
    clear Yc Ec rowsC;
  catch err_nw
    % One over-ambitious worker count must not take the remaining variants with
    % it -- that is exactly how the first attempt lost D and E.
    fprintf('[VAR]   C %2d workers FAILED (%s) — continuing.\n', nw, err_nw.message);
    delete(gcp('nocreate'));
  end
end

%% ---- D: time-major + parfor, at C's best worker count ----
if ~isnan(best_nw)
    fprintf('[VAR] D: time-major + parfor at %d workers...\n', best_nw);
    delete(gcp('nocreate'));
    parpool('Processes', best_nw);
    Ytc = parallel.pool.Constant(Yt);
    Ec  = parallel.pool.Constant(ind_event);
    tw = tic;
    nwarm = best_nw*4; warm = zeros(nwarm,1);
    parfor k = 1:nwarm
        Yw = Ytc.Value; Ew = Ec.Value;
        warm(k) = Yw(1,1) + double(Ew(1,1));
    end
    warm_s = toc(tw);

    rowsD = zeros(n_probe, T);
    tD = tic;
    parfor k = 1:n_probe
        warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:SingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
        px    = probe(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood)
            rowsD(k,:) = 0;
        else
            Yw = Ytc.Value; Ew = Ec.Value;
            tmp_ind = ~Ew(px, 2:end);
            Xt    = Yw(tmp_ind, ind_nhood);
            yt    = Yw(tmp_ind, px);
            tmpXX = Xt'*Xt;
            tmpXy = Xt'*yt;
            w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
            rowsD(k,:) = (Yw(:, ind_nhood)*w).';
        end
    end
    sD = toc(tD);
    mD = max(abs(rowsD(:) - rowsA(:)));
    res(end+1) = struct('name',sprintf('D time-major + parfor, %d workers', best_nw), ...
        'loop_s',sD,'prep_s',prepB+warm_s,'speedup',sA/sD,'max_abs',mD, ...
        'rel',mD/max(scaleA,eps), ...
        'note',sprintf('transpose %.1f s + warm %.1f s', prepB, warm_s));
    fprintf('[VAR]   D: %.1f s (%.2fx), max|diff| %.3g\n', sD, sA/sD, mD);
    clear Ytc Ec rowsD;
end
delete(gcp('nocreate'));

%% ---- E: time-major + frame subsampling for the fit only ----
fprintf('[VAR] E: time-major, fit on every %dth frame...\n', frame_skip);
rowsE = zeros(n_probe, T);
tE = tic;
for k = 1:n_probe
    px    = probe(k);
    valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
    ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
    if isempty(ind_nhood); continue; end
    keep  = false(1, T-1);
    keep(1:frame_skip:end) = true;
    tmp_ind = ~ind_event(px, 2:end) & keep;
    Xt    = Yt(tmp_ind, ind_nhood);
    yt    = Yt(tmp_ind, px);
    tmpXX = Xt'*Xt;
    tmpXy = Xt'*yt;
    w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
    rowsE(k,:) = (Yt(:, ind_nhood)*w).';
end
sE = toc(tE);
mE = max(abs(rowsE(:) - rowsA(:)));
% relative error in the reconstruction, which is what actually propagates
relE = norm(rowsE(:) - rowsA(:)) / max(norm(rowsA(:)), eps);
res(end+1) = struct('name',sprintf('E time-major + fit every %dth frame', frame_skip), ...
    'loop_s',sE,'prep_s',prepB,'speedup',sA/sE,'max_abs',mE, ...
    'rel',mE/max(scaleA,eps), ...
    'note',sprintf('NOT round-off: relative L2 error %.3g', relE));
fprintf('[VAR]   E: %.1f s (%.2fx), max|diff| %.3g, relL2 %.3g\n', sE, sA/sE, mE, relE);

%% ---- Report ----
loopA_full = sA * Npix / n_probe;   % current full-loop cost, extrapolated
L = {};
L{end+1} = sprintf('localBG variant race — %s', session_nm);
L{end+1} = datestr(now, 'yyyy-mm-dd HH:MM:SS');
L{end+1} = sprintf('%d x %d x %d; grid %d x %d (%d px); %d neighbours; probe %d px', ...
    d1, d2, T, d1s, d2s, Npix, nnz(neigh_kernel), n_probe);
L{end+1} = sprintf('client maxNumCompThreads = %d', maxNumCompThreads);
L{end+1} = sprintf('transient-suppressed samples: %.4f%%', ...
    100*nnz(ind_event)/numel(ind_event));
L{end+1} = sprintf('baseline extrapolated to all %d px: %.0f s (%.1f min)', ...
    Npix, loopA_full, loopA_full/60);
L{end+1} = '';
L{end+1} = 'variant                                          loop_s  speedup    prep_s     max|diff|      rel';
for i = 1:numel(res)
    L{end+1} = sprintf('%-46s %8.1f %8.2f %9.1f %13.3g %8.3g', ...
        res(i).name, res(i).loop_s, res(i).speedup, res(i).prep_s, ...
        res(i).max_abs, res(i).rel);  %#ok<SAGROW>
end
L{end+1} = '';
L{end+1} = 'notes:';
for i = 1:numel(res)
    if ~isempty(res(i).note)
        L{end+1} = sprintf('  %-44s %s', res(i).name, res(i).note);  %#ok<SAGROW>
    end
end
L{end+1} = '';
L{end+1} = 'Projected per-pass localBG and per-run total (2 full-res passes):';
for i = 1:numel(res)
    proj = loopA_full/res(i).speedup + res(i).prep_s + 17;   % 17 s of fixed stages
    L{end+1} = sprintf('  %-46s %6.1f min/pass   %6.1f min/run', ...
        res(i).name, proj/60, 2*proj/60);  %#ok<SAGROW>
end

report = strjoin(L, newline);
fprintf('\n%s\n', report);
out = fullfile(session_dir, 'localBG_variants.txt');
fid = fopen(out, 'w'); fprintf(fid, '%s\n', report); fclose(fid);
fprintf('\n[VAR] Report written to %s\n', out);
