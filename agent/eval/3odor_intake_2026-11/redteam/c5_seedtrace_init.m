% c5_seedtrace_init.m -- real first-pass seeds of greedyROI_endoscope at several min_corr
% (red team C5).  Initialisation stage only: loads the movie like the headless run, then
% calls initComponents_endoscope once per threshold on a fresh copy of the object, with the
% SEEDTRACE shadow copy of greedyROI_endoscope.m first on the path (caller's job).
% Caller sets: rt_nam (movie .mat), rt_mc (vector), rt_mp, rt_bd, rt_out (.mat path).
% Reads the movie only; writes only rt_out.  ASCII only.
global d1 d2 numFrame ssub tsub sframe num2read Fs neuron neuron_ds neuron_full Ybg_weights data Ysiz
global ST_SNAPS ST_SEEDS ST_STAGE
w = which('greedyROI_endoscope', '-all');
fprintf('[C5] greedyROI_endoscope -> %s\n', w{1});
assert(~isempty(strfind(w{1}, 'missed_somata_2026-11')), 'c5: the SEEDTRACE shadow is not first on the path');
nam = rt_nam; nam_mat = rt_nam;
assert(exist(nam_mat, 'file') == 2, 'c5: movie .mat not found');
data = matfile(nam_mat);
Ysiz = data.Ysiz; d1 = Ysiz(1); d2 = Ysiz(2); numFrame = Ysiz(3);
Fs = 3.75; ssub = 2; tsub = 2;
neuron_full = Sources2D('d1', d1, 'd2', d2, 'ssub', ssub, 'tsub', tsub, 'name', nam_mat, 'gSig', 9, 'gSiz', 36);
neuron_full.Fs = Fs;
neuron_full.options.search_method = 'ellipse';
neuron_full.options.dist = 3;
neuron_full.options.deconv_options = struct('type', 'ar1', 'method', 'thresholded', ...
    'optimize_pars', true, 'optimize_b', true, 'optimize_smin', false);
sframe = 1; num2read = numFrame;
cnmfe_load_data;
Y = neuron.reshape(Y, 1);
out = struct('min_corr', {}, 'min_pnr', {}, 'k', {}, 'npass', {}, 'att', {}, 'secs', {});
Cn0 = []; PNR0 = [];
for ii = 1:numel(rt_mc)
    t0 = tic;
    ST_SEEDS = {}; ST_SNAPS = {}; ST_STAGE = 'init';
    nrn = neuron.copy();
    nrn.updateParams('min_corr', rt_mc(ii), 'min_pnr', rt_mp, 'min_pixel', 36, 'bd', rt_bd);
    nrn.options.nk = 1;
    [~, Cn0, PNR0] = nrn.initComponents_endoscope(Y, [], [1, 1], false, false);
    st = ST_SEEDS{end};
    out(end + 1) = struct('min_corr', rt_mc(ii), 'min_pnr', rt_mp, 'k', st.k, 'npass', st.npass, ...
        'att', st.att, 'secs', toc(t0)); %#ok<SAGROW>
    fprintf('[C5] min_corr %.2f min_pnr %.1f: pass-1 attempts %d, initialised %d, passes %d (%.0f s)\n', ...
        rt_mc(ii), rt_mp, sum(st.att(:, 1) == 1), st.k, st.npass, toc(t0));
end
Cn0 = single(Cn0); PNR0 = single(PNR0);
save(rt_out, 'out', 'Cn0', 'PNR0');
fprintf('[C5] wrote %s\n', rt_out);
