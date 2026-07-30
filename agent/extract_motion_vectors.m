function extract_motion_vectors(session_dir, repo_root)
% Measure LOCAL MOTION directly from the raw movie at each candidate's patch,
% independent of CNMF-E's static footprint. For each candidate we estimate a
% per-frame in-plane displacement (dx,dy) of its local patch relative to the
% resting structure (Lucas-Kanade optical flow), plus a per-frame structural
% correlation to the resting patch (catches axial/z changes that are not a pure
% shift). The hypothesis: a motion artifact's "transients" coincide with the
% patch physically moving or restructuring; a real cell brightens in place while
% the patch stays put. Writes motion_vec.mat.
%
% Features per candidate (all computed over the cell's own active frames):
%   mot_mean         mean |displacement| over all frames (baseline local motion)
%   mot_p90          90th-percentile |displacement| (peak local motion here)
%   mot_active_ratio mean |disp| on active frames / mean |disp| overall
%   mot_trace_corr   corr(|disp|(t), trace(t)) -- does motion track firing?
%   mot_active_hi    fraction of active frames in the top 20% of |disp|
%   decorr_active    excess structural decorrelation (1-corr-to-rest) on active
%                    frames vs inactive -- catches z-plane restructuring
%   mot_z            frame-precise: |disp| elevation in a window AROUND transient
%                    onsets vs a circular-shift null (does lateral motion spike
%                    AT the spike?)
%   s_z              frame-precise: structural-decorrelation elevation at onsets
%                    vs the same null -- the z-motion channel the QC flag uses
%
% Reuses the same Y-load and bbox logic as extract_motion_diag.m. Read-only
% except for writing motion_vec.mat.

    addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
    rng(42);          % deterministic circular-shift null (mot_z / s_z)

    MARGIN  = 8;      % bbox margin (px) -- a bit wider, to see a neighbor drift in
    K_ACT   = 2.5;    % activity threshold in robust sigmas
    MIN_ACT = 5;      % min active frames to score a candidate

    rn     = load(fullfile(session_dir, 'review_neuron.mat'));
    neuron = rn.neuron;
    A  = full(neuron.A);
    C  = neuron.C_raw;
    d1 = double(neuron.options.d1);
    d2 = double(neuron.options.d2);
    [P, N] = size(A);

    [~, nm] = fileparts(session_dir);
    mv   = matfile(fullfile(session_dir, [nm, '.mat']));
    Ysiz = mv.Ysiz;
    Y    = mv.Y;
    if ndims(Y) == 3
        Y = reshape(Y, Ysiz(1) * Ysiz(2), Ysiz(3));
    elseif size(Y, 1) ~= P && size(Y, 2) == P
        Y = Y';
    end
    assert(size(Y, 1) == P, 'video/footprint pixel mismatch: Y=%d A=%d', size(Y,1), P);
    Y = single(Y);
    T = size(Y, 2);
    fprintf('[%s] %d candidates, %d frames\n', nm, N, T);

    fnames = {'mot_mean','mot_p90','mot_active_ratio','mot_trace_corr', ...
              'mot_active_hi','decorr_active','mot_z','s_z'};
    feats  = nan(N, numel(fnames));

    for i = 1:N
        a   = A(:, i);
        sup = find(a > 0);
        if numel(sup) < 5; continue; end
        [yy, xx] = ind2sub([d1 d2], sup);
        ys = max(1, min(yy) - MARGIN):min(d1, max(yy) + MARGIN);
        xs = max(1, min(xx) - MARGIN):min(d2, max(xx) + MARGIN);
        h = numel(ys); w = numel(xs);
        if h < 3 || w < 3; continue; end
        [Xg, Yg] = meshgrid(xs, ys);
        bb = sub2ind([d1 d2], Yg(:), Xg(:));       % bbox, pixel order = rows(y) fastest
        Ybb = double(Y(bb, :));                     % nbb x T

        ci = C(i, :);
        baseline = median(ci);
        sigma    = median(abs(ci - baseline)) / 0.6745;
        if sigma < 1e-9; continue; end
        thr    = baseline + K_ACT * sigma;
        active = ci > thr;
        if sum(active) < MIN_ACT || sum(~active) < MIN_ACT; continue; end

        % Resting structure = mean over inactive frames (cell silent).
        ref     = mean(Ybb(:, ~active), 2);          % nbb x 1
        ref_img = reshape(ref, h, w);
        [gx, gy] = gradient(ref_img);                % spatial gradients
        G = [gx(:), gy(:)];                          % nbb x 2
        AtA = G' * G;
        if rcond(AtA) < 1e-6; continue; end          % flat/degenerate patch

        % Lucas-Kanade per-frame rigid displacement (vectorised over T):
        It  = Ybb - ref;                             % nbb x T (deviation from rest)
        D   = AtA \ (-(G' * It));                    % 2 x T  -> [dx; dy]
        mot = sqrt(D(1, :) .^ 2 + D(2, :) .^ 2);     % 1 x T local motion magnitude

        % Per-frame structural correlation to the resting patch.
        refc = ref - mean(ref); nref = sqrt(sum(refc .^ 2));
        Ybbc = Ybb - mean(Ybb, 1);
        pc   = (refc' * Ybbc) ./ (sqrt(sum(Ybbc .^ 2, 1)) * nref + 1e-9);  % 1 x T

        mot_mean   = mean(mot);
        mot_p90    = quantile_p(mot, 0.90);
        mot_act    = mean(mot(active)) / (mot_mean + 1e-9);
        rr         = corrcoef(mot(:), ci(:)); mot_tc = rr(1, 2);
        hi_thr     = quantile_p(mot, 0.80);
        mot_hi     = mean(mot(active) > hi_thr);
        decorr_act = mean(1 - pc(active)) - mean(1 - pc(~active));

        % Frame-precise onset coincidence: does motion / structure spike right AT
        % each transient onset? Onsets = rising crossings of baseline + K_ACT*sigma
        % (refractory 4 frames so one long transient is not counted many times).
        onsets = find(ci(2:end) > thr & ci(1:end-1) <= thr) + 1;
        if ~isempty(onsets)
            kp = onsets(1);
            for oo = onsets(2:end)
                if oo - kp(end) >= 4; kp(end+1) = oo; end %#ok<AGROW>
            end
            onsets = kp;
        end
        if numel(onsets) >= 3
            mot_z_v = onset_coince(mot,    onsets, T);
            s_z_v   = onset_coince(1 - pc, onsets, T);
        else
            mot_z_v = NaN; s_z_v = NaN;
        end

        feats(i, :) = [mot_mean, mot_p90, mot_act, mot_tc, mot_hi, decorr_act, ...
                       mot_z_v, s_z_v];
    end

    feature_names = fnames; %#ok<NASGU>
    save(fullfile(session_dir, 'motion_vec.mat'), 'feats', 'feature_names', 'N');
    fprintf('[%s] motion_vec.mat saved (%d/%d scored)\n', nm, sum(~isnan(feats(:,1))), N);
end


function q = quantile_p(v, p)
% Simple percentile without the Statistics toolbox.
    v = sort(v(:));
    if isempty(v); q = NaN; return; end
    idx = max(1, min(numel(v), round(p * numel(v))));
    q = v(idx);
end


function z = onset_coince(x, onsets, T)
% Elevation of the per-frame series x in a [-1,+2]-frame window around each
% transient onset vs a CIRCULAR-SHIFT NULL (shift x's phase against the fixed
% onsets N times). Returns a per-cell z-score; the null controls for the
% "busy session" confound (a globally high-motion patch also has a high null).
% Mirrors test_motion_onset.py exactly (window, refractory, N=200).
    W  = -1:2;
    NN = 200;
    x  = x(:).';                                  % 1 x T row
    onsets = onsets(:);                           % col
    base = median(x, 'omitnan');
    bidx = mod((onsets + W) - 1, T) + 1;          % n_on x wlen (circular, 1-based)
    obs  = mean(max(x(bidx), [], 2), 'omitnan') - base;
    shifts = randi(T - 1, NN, 1);
    nullv  = zeros(NN, 1);
    for j = 1:NN
        idx = mod((onsets + W + shifts(j)) - 1, T) + 1;
        nullv(j) = mean(max(x(idx), [], 2), 'omitnan') - base;
    end
    z = (obs - mean(nullv)) / (std(nullv) + 1e-9);
end
