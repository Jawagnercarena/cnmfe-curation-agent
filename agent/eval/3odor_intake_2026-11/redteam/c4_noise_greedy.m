% c4_noise_greedy.m -- the REAL greedyROI_endoscope on a pure-noise movie (red team C4).
% Caller sets: rt_T (frames at half rate), rt_seed, rt_mc (vector of min_corr), rt_mp
% (scalar min_pnr), rt_out (.mat path, folder must exist), rt_pick (true: also run the
% pickNeurons stage on the same noise).  cnmfe_setup has been run by the caller.
% Noise: Gaussian, white in time, lag-1 spatial correlation 0.18 (columns) / 0.13 (rows),
% built with 3-tap kernels like the Python fields.  Writes only rt_out.  ASCII only.
rng(rt_seed);
d1s = 256; d2s = 256;
ax = (1 - sqrt(1 - 2 * 0.18^2)) / (2 * 0.18);
ay = (1 - sqrt(1 - 2 * 0.13^2)) / (2 * 0.13);
Y = randn(d1s, d2s, rt_T);
Y = imfilter(Y, [ax, 1, ax], 'circular');
Y = imfilter(Y, [ay; 1; ay], 'circular');
Y = reshape(Y, d1s * d2s, rt_T);
base = Sources2D('d1', d1s, 'd2', d2s, 'ssub', 1, 'tsub', 1, 'gSig', 5, 'gSiz', 18);
base.Fs = 3.75 / 2;
base.options.search_method = 'ellipse';
base.options.dist = 3;
base.options.deconv_options = struct('type', 'ar1', 'method', 'thresholded', ...
    'optimize_pars', true, 'optimize_b', true, 'optimize_smin', false);
fprintf('[C4] greedyROI_endoscope -> %s\n', which('greedyROI_endoscope'));
res = struct('T', {}, 'seed', {}, 'min_corr', {}, 'min_pnr', {}, 'k_init', {}, 'k_pick', {}, 'secs', {});
Cn0 = []; PNR0 = [];
for ii = 1:numel(rt_mc)
    t0 = tic;
    nrn = base.copy();
    nrn.updateParams('min_corr', rt_mc(ii), 'min_pnr', rt_mp, 'min_pixel', 36, 'bd', 10);
    nrn.options.nk = 1;
    [~, Cn0, PNR0] = nrn.initComponents_endoscope(Y, [], [1, 1], false, false);
    k_init = size(nrn.A, 2);
    k_pick = NaN;
    if rt_pick
        nrn2 = base.copy();
        nrn2.updateParams('min_corr', rt_mc(ii), 'min_pnr', rt_mp, 'min_pixel', 36, 'bd', 10);
        nrn2.options.nk = 1;
        nrn2.A = sparse(d1s * d2s, 0); nrn2.C = zeros(0, rt_T); nrn2.C_raw = zeros(0, rt_T); nrn2.S = zeros(0, rt_T);
        nrn2.P.kernel_pars = [];
        nrn2.pickNeurons(Y, [1, 1], 'auto', false);
        k_pick = size(nrn2.A, 2);
    end
    res(end + 1) = struct('T', rt_T, 'seed', rt_seed, 'min_corr', rt_mc(ii), 'min_pnr', rt_mp, ...
        'k_init', k_init, 'k_pick', k_pick, 'secs', toc(t0)); %#ok<SAGROW>
    fprintf('[C4] T %d seed %d min_corr %.3f min_pnr %.1f: init components %d, pick components %d (%.0f s)\n', ...
        rt_T, rt_seed, rt_mc(ii), rt_mp, k_init, k_pick, toc(t0));
end
Cn0 = single(Cn0); PNR0 = single(PNR0);
save(rt_out, 'res', 'Cn0', 'PNR0');
fprintf('[C4] wrote %s\n', rt_out);
