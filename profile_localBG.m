%% profile_localBG.m
% Where do the 40+ minutes of the full-resolution background estimate go, and
% does the per-pixel ring regression actually parallelise?
%
% Runs against an already-finished session (needs neuron.mat plus the raw data
% .mat).  Nothing the pipeline depends on is touched: the only thing written is
% a text report, localBG_profile.txt, in the session folder.
%
% Usage -- the machine should be otherwise idle, see the memory note below:
%   session_dir   = 'D:\Julian_CNMFe\BLA\3odor\AVG5x-TSeries-07082026-bla36-810um-25z-000';
%   n_probe       = 2000;         % optional: pixels sampled for the loop timing
%   worker_counts = [6 8 12];     % optional: pool sizes to test
%   run('C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\profile_localBG.m')
%
% The stages below mirror local_background.m statement for statement (including
% its quirks -- see the centering note), so the extrapolated loop cost is
% comparable with the "Time cost in estimating the background" line the headless
% run prints: 2605.93 s for bla36-810um, 2654.98 s for bla37-230um.
%
% MEMORY: this script drops the caller's full-resolution Y as soon as the
% residual is formed, so its peak sits ~27 GB below what cnmfe_update_BG really
% holds -- and ~54 GB below the peak after Ysignal = Y - Ybg.  Add that back
% when reading the per-worker figures.

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
% cvx ships pre-R2013a shims (lib/narginchk_) that shadow MATLAB builtins and
% warn on every launch; cvx_startup skips them on modern MATLAB, genpath cannot.
cvx_dirs = strsplit(genpath(fullfile(repo_root, 'cvx')), pathsep);
cvx_dirs = cvx_dirs(~cellfun(@isempty, cvx_dirs));
addpath(strjoin(cvx_dirs(~endsWith(cvx_dirs, [filesep 'narginchk_'])), pathsep));
clear cvx_dirs;
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('profile_localBG: session_dir must be set before running.');
end
if ~exist('n_probe', 'var')       || isempty(n_probe);       n_probe = 2000;         end
if ~exist('worker_counts', 'var') || isempty(worker_counts); worker_counts = [6 8 12]; end

s = struct();   % stage timings, seconds

%% Load the finished session
fprintf('[PROFILE] Loading neuron.mat...\n');
ndata  = load(fullfile(session_dir, 'neuron.mat'));
neuron = ndata.neuron;
clear ndata;

% Locate the raw data .mat (same exclusion list as CNMFe_precompute_BG.m).
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
    if isempty(cand)
        error('[PROFILE] Could not find the raw data .mat in %s', session_dir);
    end
    nam_mat = fullfile(session_dir, cand(1).name);
end

data_file = matfile(nam_mat);
Ysiz = data_file.Ysiz;
d1 = Ysiz(1); d2 = Ysiz(2); T = Ysiz(3);
fprintf('[PROFILE] %s: %d x %d x %d frames\n', session_nm, d1, d2, T);

if ~isfield(neuron.P, 'sn') || isempty(neuron.P.sn)
    error('[PROFILE] neuron.P.sn is empty — this session predates the noise estimate.');
end

fprintf('[PROFILE] Loading full-resolution data (~%.1f GB as double)...\n', ...
    d1*d2*T*8/2^30);
t = tic;
Y = double(data_file.Y);
Y = reshape(Y, d1*d2, T);
s.load_raw = toc(t);

%% Stage 1 — the residual cnmfe_update_BG hands to localBG
t = tic;
Yb = Y - neuron.A*neuron.C;
s.residual = toc(t);
clear Y;    % see the MEMORY note in the header

% Parameters exactly as cnmfe_update_BG sets them.
ssub   = 2;                                    % spatial_ds_factor
rr0    = ceil(neuron.options.gSiz * 1.5);      % bg_neuron_ratio = 1.5
thresh = 10;
sn     = neuron.reshape(neuron.P.sn, 2);
fprintf('[PROFILE] gSiz=%d -> rr=%d, ssub=%d, thresh=%d\n', ...
    neuron.options.gSiz, rr0, ssub, thresh);

Yb = reshape(Yb, d1, d2, T);

%% Stage 2 — centering (local_background.m:25-26)
% Note the original's quirk: bsxfun(@minus, Ymean, ones(1,1,T)) expands to
% Ymean-1 per frame, so this both centers AND adds a constant 1.  Replicated
% verbatim so the timing and the numbers match production.
t = tic;
Ymean = mean(Yb, 3);
Yb    = Yb - bsxfun(@minus, Ymean, ones(1, 1, T));
s.centering = toc(t);

%% Stage 3 — spatial downsample (line 39)
t = tic;
Yb   = imresize(Yb, 1./ssub);
[d1s, d2s, ~] = size(Yb);
rr   = round(rr0/ssub) + 1;
sn_s = imresize(sn, 1./ssub);
s.resize_down = toc(t);
fprintf('[PROFILE] downsampled grid: %d x %d (%d pixels), ring radius %d\n', ...
    d1s, d2s, d1s*d2s, rr);

%% Stage 4 — ring filter (lines 66-72).  57x57 kernel over every frame.
t = tic;
rsub_k = (-rr):rr;
[cind, rind]  = meshgrid(rsub_k, rsub_k);
R             = sqrt(cind.^2 + rind.^2);
neigh_kernel  = (R >= rr) .* (R < rr+1);
Yconv = bsxfun(@times, imfilter(Yb, neigh_kernel), ...
    1./imfilter(ones(d1s, d2s), neigh_kernel));
s.ring_filter = toc(t);
fprintf('[PROFILE] ring kernel %dx%d, %d neighbours per pixel\n', ...
    size(neigh_kernel,1), size(neigh_kernel,2), nnz(neigh_kernel));

%% Stage 5 — transient suppression (lines 73-75)
% This is what reconstructBG does NOT do: samples more than thresh*sn above the
% ring mean are replaced by the ring mean before the fit and the reconstruction.
t = tic;
ind_event = (bsxfun(@times, Yb - Yconv, 1./sn_s) > thresh);
Yb(ind_event) = Yconv(ind_event);
ind_event = reshape(ind_event, d1s*d2s, []);
s.event_mask = toc(t);
clear Yconv;
fprintf('[PROFILE] samples suppressed as transients: %.3f%%\n', ...
    100*nnz(ind_event)/numel(ind_event));

%% Stage 6 — neighbour index construction (lines 87-99)
t = tic;
[r_shift, c_shift] = find(neigh_kernel);
r_shift = r_shift - rr - 1;
c_shift = c_shift - rr - 1;
[csub, rsub] = meshgrid(1:d2s, 1:d1s);
csub = reshape(csub, [], 1);
rsub = reshape(rsub, [], 1);
csub = bsxfun(@plus, csub, c_shift');
rsub = bsxfun(@plus, rsub, r_shift');
oob  = or(or(csub<1, csub>d2s), or(rsub<1, rsub>d1s));
csub(oob) = nan;
rsub(oob) = nan;
s.neighbour_index = toc(t);

Yb   = reshape(Yb, d1s*d2s, []);
Npix = d1s*d2s;

%% Stage 7 — the pixel loop, measured on a random probe and extrapolated
% Random sampling (rather than a contiguous block) so boundary pixels, which
% have fewer valid neighbours and are cheaper, appear in proportion.
rng(0);
n_probe = min(n_probe, Npix);
probe   = sort(randperm(Npix, n_probe));

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');
warning('off', 'MATLAB:singularMatrix');

fprintf('[PROFILE] serial loop over %d of %d pixels...\n', n_probe, Npix);
rows_ser = zeros(n_probe, size(Yb, 2));
t = tic;
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
    rows_ser(k,:) = w'*Yb(ind_nhood,:);
end
s.loop_probe_serial = toc(t);
loop_total_est = s.loop_probe_serial * Npix / n_probe;

%% Stage 8 — upsample and rebaseline (lines 149-159), timed on a full-size stub
% Uses rows already computed for the probe is not meaningful here, so this times
% the two array operations on a correctly sized zero array: the cost is shape-
% driven, not value-driven.
t = tic;
Yest_stub = zeros(d1s, d2s, size(Yb,2));
Yest_stub = imresize(Yest_stub, [d1, d2]);
Ybase     = Ymean - mean(Yest_stub, 3);
Yest_stub = bsxfun(@plus, Yest_stub, Ybase);
s.resize_up_baseline = toc(t);
clear Yest_stub Ybase;

%% Parallel scaling on the same probe
per_worker_gb = (numel(Yb)*8 + numel(ind_event) + numel(rsub)*8*2) / 2^30;
try
    mm      = memory;
    free_gb = mm.MemAvailableAllArrays / 2^30;
catch
    free_gb = NaN;
end
fprintf('[PROFILE] ~%.1f GB per worker; %.0f GB reported available\n', ...
    per_worker_gb, free_gb);

scal = struct('workers', {}, 'pool_s', {}, 'broadcast_s', {}, ...
              'loop_s', {}, 'speedup', {}, 'max_abs_diff', {}, 'rel_diff', {});
nthreads_worker = NaN;

for nw = worker_counts
    if ~isnan(free_gb) && nw*per_worker_gb > 0.60*free_gb
        fprintf('[PROFILE] SKIPPING %d workers: %.0f GB needed vs %.0f GB available.\n', ...
            nw, nw*per_worker_gb, free_gb);
        continue;
    end
    delete(gcp('nocreate'));

    fprintf('[PROFILE] starting pool of %d...\n', nw);
    t = tic; parpool('Processes', nw); pool_s = toc(t);

    % parallel.pool.Constant sends the big arrays once instead of re-broadcasting
    % per parfor statement -- this is also how a real implementation should do it,
    % so the cost measured here is the cost production would pay.
    t = tic;
    Yc = parallel.pool.Constant(Yb);
    Ec = parallel.pool.Constant(ind_event);
    broadcast_s = toc(t);

    if isnan(nthreads_worker)
        try
            f = parfevalOnAll(@maxNumCompThreads, 1);
            nt = fetchOutputs(f);
            nthreads_worker = nt(1);
        catch
            nthreads_worker = NaN;
        end
    end

    rows_par = zeros(n_probe, size(Yb, 2));
    t = tic;
    parfor k = 1:n_probe
        % Client-side warning state does NOT reach workers, so suppress here or
        % the log fills with nearlySingularMatrix.
        warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:SingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
        px    = probe(k);
        valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
        ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
        if isempty(ind_nhood)
            rows_par(k,:) = 0;
        else
            Yw    = Yc.Value;
            tmp_ind = ~Ec.Value(px, 2:end);
            X     = Yw(ind_nhood, tmp_ind);
            y     = Yw(px, tmp_ind);
            tmpXX = X*X';
            tmpXy = X*y';
            w     = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
            rows_par(k,:) = w'*Yw(ind_nhood,:);
        end
    end
    loop_s = toc(t);

    max_abs = max(abs(rows_par(:) - rows_ser(:)));
    scale   = max(abs(rows_ser(:)));
    rel     = max_abs / max(scale, eps);

    scal(end+1) = struct('workers', nw, 'pool_s', pool_s, ...
        'broadcast_s', broadcast_s, 'loop_s', loop_s, ...
        'speedup', s.loop_probe_serial/loop_s, ...
        'max_abs_diff', max_abs, 'rel_diff', rel);   %#ok<SAGROW>

    fprintf('[PROFILE]   %2d workers: loop %.1f s (%.2fx), broadcast %.1f s, max|diff| %.3g (rel %.3g)\n', ...
        nw, loop_s, s.loop_probe_serial/loop_s, broadcast_s, max_abs, rel);
    clear Yc Ec rows_par;
end
delete(gcp('nocreate'));

%% Report
stage_names = {'load_raw','residual','centering','resize_down','ring_filter', ...
               'event_mask','neighbour_index','resize_up_baseline'};
fixed_total = 0;
for i = 1:numel(stage_names)
    if isfield(s, stage_names{i}); fixed_total = fixed_total + s.(stage_names{i}); end
end
% load_raw is the caller's cost, not localBG's — report it but exclude it from
% the localBG total so the number is comparable with the headless log line.
localbg_fixed = fixed_total - s.load_raw;
localbg_total = localbg_fixed + loop_total_est;

L = {};
L{end+1} = sprintf('localBG profile — %s', session_nm);
L{end+1} = sprintf('%s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
L{end+1} = sprintf('%d x %d x %d frames; downsampled grid %d x %d (%d px); ring radius %d, %d neighbours', ...
    d1, d2, T, d1s, d2s, Npix, rr, nnz(neigh_kernel));
L{end+1} = sprintf('client maxNumCompThreads = %d; worker maxNumCompThreads = %g', ...
    maxNumCompThreads, nthreads_worker);
L{end+1} = '';
L{end+1} = 'Stage breakdown (seconds):';
for i = 1:numel(stage_names)
    nm = stage_names{i};
    if isfield(s, nm)
        L{end+1} = sprintf('  %-22s %9.1f', nm, s.(nm));  %#ok<SAGROW>
    end
end
L{end+1} = sprintf('  %-22s %9.1f   (%d of %d pixels)', 'pixel loop (probe)', ...
    s.loop_probe_serial, n_probe, Npix);
L{end+1} = '';
L{end+1} = sprintf('Extrapolated pixel loop, all %d pixels: %.0f s (%.1f min)', ...
    Npix, loop_total_est, loop_total_est/60);
L{end+1} = sprintf('localBG total excl. raw load:          %.0f s (%.1f min)', ...
    localbg_total, localbg_total/60);
L{end+1} = sprintf('  of which the pixel loop is          %.1f%%', ...
    100*loop_total_est/localbg_total);
L{end+1} = sprintf('  fixed (non-loop) stages             %.0f s (%.1f min)', ...
    localbg_fixed, localbg_fixed/60);
L{end+1} = '';
L{end+1} = sprintf('Parallel scaling (same %d-pixel probe, ~%.1f GB per worker):', ...
    n_probe, per_worker_gb);
if isempty(scal)
    L{end+1} = '  no worker counts were run (all skipped for memory).';
else
    L{end+1} = '  workers   loop_s  speedup  broadcast_s  pool_s   max|diff|    rel_diff';
    for i = 1:numel(scal)
        L{end+1} = sprintf('  %7d %8.1f %8.2f %12.1f %7.1f %11.3g %11.3g', ...
            scal(i).workers, scal(i).loop_s, scal(i).speedup, ...
            scal(i).broadcast_s, scal(i).pool_s, ...
            scal(i).max_abs_diff, scal(i).rel_diff);  %#ok<SAGROW>
    end
    best = scal(1);
    for i = 2:numel(scal)
        if scal(i).loop_s < best.loop_s; best = scal(i); end
    end
    proj = localbg_fixed + loop_total_est/best.speedup + best.broadcast_s;
    L{end+1} = '';
    L{end+1} = sprintf('Projected localBG at %d workers: %.0f s (%.1f min) vs %.0f s (%.1f min) now', ...
        best.workers, proj, proj/60, localbg_total, localbg_total/60);
    L{end+1} = sprintf('Two passes per headless run: %.1f min -> %.1f min', ...
        2*localbg_total/60, 2*proj/60);
    L{end+1} = '';
    L{end+1} = sprintf(['rel_diff is the parfor-vs-serial divergence: workers run single-threaded\n' ...
        'BLAS while the client multithreads X*X'', so the summation order differs.\n' ...
        'Expect ~1e-12; anything larger than ~1e-8 means something else is wrong.']);
end

report = strjoin(L, newline);
fprintf('\n%s\n', report);
out = fullfile(session_dir, 'localBG_profile.txt');
fid = fopen(out, 'w');
fprintf(fid, '%s\n', report);
fclose(fid);
fprintf('\n[PROFILE] Report written to %s\n', out);
