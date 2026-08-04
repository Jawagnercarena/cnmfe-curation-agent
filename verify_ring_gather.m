%% verify_ring_gather.m
% Verifies the two changes made to the reconstructBG fast path in
% CNMFe_final_save.m, on real session data:
%
%   A  original     pixel-major serial : Ybg(m,:) = w(2,:) * Yds(w(1,:), :)
%   B  shipped      time-major  serial : M = Ydst(:,w(1,:)).';  w(2,:)*M
%   C  candidate    time-major  parfor : same body, thread pool
%
% Claim 1 (already shipped): B is bit-identical to A.  A transpose is a pure
%   permutation, so M holds the same bytes, and the multiply is then the same
%   BLAS call on the same operands.
% Claim 2 (unverified): C is bit-identical to B.  parfor does not alter the
%   arithmetic inside an iteration and the rows are independent.
%
% A is ~5x slower than B, so A runs on a subset of pixels (boundary rings have
% fewer neighbours than interior ones, so the subset deliberately spans both
% ends of the index range).  B and C run over every pixel.
%
% Read-only w.r.t. the session folder; the report goes to report_dir.
%
%   session_dir = 'D:\Julian_CNMFe\BLA\3odor\AVG5x-TSeries-07082026-bla36-810um-25z-000';
%   report_dir  = '<scratch>';
%   run('C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\verify_ring_gather.m')

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('verify_ring_gather: session_dir must be set.');
end
if ~exist('report_dir', 'var') || isempty(report_dir)
    error('verify_ring_gather: report_dir must be set.');
end

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');

%% ---- rebuild Yds exactly as the reviewer path does ----
fprintf('[VERIFY] Loading review set + cached weights...\n');
rdata   = load(fullfile(session_dir, 'review_neuron.mat'));
neuron  = rdata.neuron; clear rdata;
wdata   = load(fullfile(session_dir, 'Ybg_weights.mat'));
weights = wdata.Ybg_weights; clear wdata;

[~, session_nm] = fileparts(session_dir);
nam_mat = fullfile(session_dir, [session_nm, '.mat']);
if ~isfile(nam_mat)
    cand   = dir(fullfile(session_dir, '*.mat'));
    reject = {'neuron','Cn','Coor','pnr','Ybg','Ybg_mean','spatial_footprints', ...
              'review_neuron','Ybg_weights','review_checkpoint'};
    for i = numel(cand):-1:1
        [~, nm] = fileparts(cand(i).name);
        if any(strcmp(nm, reject)); cand(i) = []; end
    end
    nam_mat = fullfile(session_dir, cand(1).name);
end
data_file = matfile(nam_mat);
Ysiz = data_file.Ysiz; d1 = Ysiz(1); d2 = Ysiz(2); T = Ysiz(3);
fprintf('[VERIFY] %s: %d x %d x %d, %d review neurons\n', ...
    session_nm, d1, d2, T, size(neuron.C, 1));

t = tic;
Y    = reshape(double(data_file.Y), d1*d2, T);
fprintf('[VERIFY]   load            %6.1f s\n', toc(t));
t = tic;
Yres = reshape(Y - neuron.A * neuron.C, d1, d2, T);
clear Y;
fprintf('[VERIFY]   residual Y-A*C  %6.1f s\n', toc(t));
t = tic;
dims = weights.dims;
b0   = mean(Yres, 3);
Yds  = imresize(bsxfun(@minus, Yres, b0), dims);
clear Yres b0;
Yds  = reshape(Yds, [], T);
nPix = size(Yds, 1);
fprintf('[VERIFY]   demean+downsamp %6.1f s   -> Yds %d x %d (%.1f GB)\n', ...
    toc(t), nPix, T, numel(Yds)*8/2^30);

Wcell = weights.weights;
J     = cellfun(@(c) size(c, 2), Wcell(:));
fprintf('[VERIFY] ring size J: min %d  median %d  max %d\n', ...
    min(J), median(J), max(J));

%% ---- B: time-major serial (what is shipped) ----
t = tic; Ydst = Yds.'; s_tr = toc(t);
fprintf('[VERIFY] transpose to time-major %.1f s\n', s_tr);

Ybg_B = zeros(nPix, T);
t = tic;
for m = 1:nPix
    w = Wcell{m};
    M = Ydst(:, w(1,:)).';
    Ybg_B(m, :) = w(2,:) * M;
end
s_B = toc(t);
fprintf('[VERIFY] B time-major serial     %7.1f s (%.1f min)\n', s_B, s_B/60);

%% ---- C: time-major parfor on a THREAD pool ----
n_thr = 0;
try
    if ~isempty(gcp('nocreate')); delete(gcp('nocreate')); end
    tp    = parpool('Threads');
    n_thr = tp.NumWorkers;
catch ME
    fprintf('[VERIFY] *** parpool(''Threads'') FAILED: %s / %s\n', ...
        ME.identifier, ME.message);
end

Ybg_C  = zeros(nPix, T);
s_C    = NaN;
par_ok = false;
if n_thr > 0
    fprintf('[VERIFY] thread pool: %d workers\n', n_thr);
    t = tic;
    try
        parfor m = 1:nPix
            w = Wcell{m};
            M = Ydst(:, w(1,:)).';
            Ybg_C(m, :) = w(2,:) * M;
        end
        par_ok = true;
    catch ME
        fprintf('[VERIFY] *** parfor body FAILED: %s / %s\n', ...
            ME.identifier, ME.message);
    end
    s_C = toc(t);
    fprintf('[VERIFY] C time-major parfor     %7.1f s (%.1f min)\n', s_C, s_C/60);
end

%% ---- A: original pixel-major serial, on a subset ----
% Span both ends of the index range so short boundary rings and full interior
% rings are both covered.
idx = unique([1:2000, (nPix-1999):nPix, round(linspace(1, nPix, 6000))]);
Ybg_A = zeros(numel(idx), T);
t = tic;
for k = 1:numel(idx)
    w = Wcell{idx(k)};
    Ybg_A(k, :) = w(2,:) * Yds(w(1,:), :);
end
s_A = toc(t);
fprintf('[VERIFY] A pixel-major serial    %7.1f s for %d px (%.0f s extrapolated)\n', ...
    s_A, numel(idx), s_A * nPix / numel(idx));
clear Yds Ydst;

%% ---- compare ----
dBC = abs(Ybg_B - Ybg_C);
mBC = max(dBC(:)); nBC = nnz(dBC); clear dBC;
dAB = abs(Ybg_A - Ybg_B(idx, :));
mAB = max(dAB(:)); nAB = nnz(dAB); clear dAB;

L = {};
L{end+1} = sprintf('ring gather verification -- %s', session_nm);
L{end+1} = datestr(now, 'yyyy-mm-dd HH:MM:SS');
L{end+1} = sprintf('%d x %d x %d; dims [%d %d]; %d pixels; J min/med/max %d/%d/%d', ...
    d1, d2, T, dims(1), dims(2), nPix, min(J), median(J), max(J));
L{end+1} = '';
L{end+1} = 'timings:';
L{end+1} = sprintf('  A pixel-major serial   %8.1f s  (extrapolated from %d px)', ...
    s_A * nPix / numel(idx), numel(idx));
L{end+1} = sprintf('  B time-major  serial   %8.1f s  (+%.1f s transpose)', s_B, s_tr);
L{end+1} = sprintf('  C time-major  parfor   %8.1f s  on %d threads', s_C, n_thr);
L{end+1} = '';
L{end+1} = sprintf('  B vs A speedup         %8.2fx', (s_A * nPix / numel(idx)) / s_B);
L{end+1} = sprintf('  C vs B speedup         %8.2fx', s_B / s_C);
L{end+1} = sprintf('  C vs A speedup         %8.2fx', (s_A * nPix / numel(idx)) / s_C);
L{end+1} = '';
L{end+1} = 'bit-exactness:';
L{end+1} = sprintf('  B vs A  (%d px x %d frames = %d values)   max|diff| %.3g   differing %d', ...
    numel(idx), T, numel(idx)*T, mAB, nAB);
L{end+1} = sprintf('  C vs B  (%d px x %d frames = %d values)   max|diff| %.3g   differing %d', ...
    nPix, T, nPix*T, mBC, nBC);
L{end+1} = '';
if ~par_ok
    L{end+1} = 'VERDICT: *** parfor did NOT run -- see log above ***';
elseif mBC == 0 && nBC == 0 && mAB == 0 && nAB == 0
    L{end+1} = 'VERDICT: BIT-IDENTICAL -- the thread-pool parfor and the shipped';
    L{end+1} = '         time-major serial loop both return exactly what the original';
    L{end+1} = '         pixel-major loop returned.';
else
    L{end+1} = 'VERDICT: *** DIFFERENCES FOUND -- DO NOT DEPLOY ***';
end

txt = strjoin(L, newline);
fprintf('\n%s\n', txt);
out = fullfile(report_dir, 'ring_gather_verify.txt');
fid = fopen(out, 'w'); fprintf(fid, '%s\n', txt); fclose(fid);
fprintf('[VERIFY] Report written to %s\n', out);

if n_thr > 0; delete(gcp('nocreate')); end
