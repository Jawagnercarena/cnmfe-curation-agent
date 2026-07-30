function extract_motion_diag(session_dir, repo_root)
% Backfill a windowed footprint-vs-video spatial-stability feature set for every
% review candidate in a session, to test whether motion artifacts -- cells that
% only occupy their footprint until brain motion (a z-plane cell that pops in, or
% a neighbor that drifts into the outline) -- are separable from real cells by how
% stably the footprint matches the actual video over time. Writes motion_diag.mat.
%
% For each candidate, over a local bounding box around its footprint:
%   - find the frames where its trace is active (robust threshold)
%   - split the recording into K time windows
%   - in each window, build the mean activity image over that window's active
%     frames and correlate it spatially with the static footprint
% A real cell matches its footprint consistently in every window; a motion
% artifact matches only while it is "really there", so the per-window correlation
% drops or its activity centroid wanders.
%
% Features per candidate:
%   stab_global    corr(footprint, activity image over ALL active frames)
%   stab_mean      mean per-window spatial correlation
%   stab_min       worst per-window correlation (real cells stay high)
%   stab_std       spread of per-window correlation (motion is unstable)
%   stab_drop      stab_global - stab_min (how far the worst window falls)
%   centroid_drift max distance (px) between per-window activity centroids
%   valid_win_frac fraction of windows with enough activity to score
%
% Read-only except for writing motion_diag.mat into session_dir.

    addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));

    K       = 8;      % time windows
    MARGIN  = 6;      % bbox margin (px) around the footprint support
    K_ACT   = 2.5;    % activity threshold in robust sigmas
    MIN_ACT = 3;      % min active frames to score a window

    rn     = load(fullfile(session_dir, 'review_neuron.mat'));
    neuron = rn.neuron;
    A  = full(neuron.A);              % P x N
    C  = neuron.C_raw;                % N x T
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
    fprintf('[%s] %d candidates, %d pixels, %d frames\n', nm, N, P, T);

    fnames = {'stab_global','stab_mean','stab_min','stab_std', ...
              'stab_drop','centroid_drift','valid_win_frac'};
    feats  = nan(N, numel(fnames));
    edges  = round(linspace(1, T + 1, K + 1));

    for i = 1:N
        a   = A(:, i);
        sup = find(a > 0);
        if numel(sup) < 5; continue; end

        [yy, xx] = ind2sub([d1 d2], sup);
        ys = max(1, min(yy) - MARGIN):min(d1, max(yy) + MARGIN);
        xs = max(1, min(xx) - MARGIN):min(d2, max(xx) + MARGIN);
        [Xg, Yg]   = meshgrid(xs, ys);
        bb         = sub2ind([d1 d2], Yg(:), Xg(:));   % bbox linear indices
        [byy, bxx] = ind2sub([d1 d2], bb);
        abb   = double(a(bb));
        abb_c = abb - mean(abb);
        na    = sqrt(sum(abb_c .^ 2));
        if na < 1e-9; continue; end

        Ybb = double(Y(bb, :));                          % nbb x T
        ci  = C(i, :);
        baseline = median(ci);
        sigma    = median(abs(ci - baseline)) / 0.6745;  % robust sigma
        if sigma < 1e-9; continue; end
        thr = baseline + K_ACT * sigma;

        active_all = ci > thr;
        if sum(active_all) < MIN_ACT; continue; end
        gimg        = mean(Ybb(:, active_all), 2) - mean(Ybb, 2);
        stab_global = local_corr(gimg, abb_c, na);

        rk  = nan(1, K);
        cen = nan(K, 2);
        for k = 1:K
            fr  = edges(k):edges(k + 1) - 1;
            act = fr(ci(fr) > thr);
            if numel(act) < MIN_ACT; continue; end
            img   = mean(Ybb(:, act), 2) - mean(Ybb(:, fr), 2);
            rk(k) = local_corr(img, abb_c, na);
            w  = max(img, 0);
            sw = sum(w);
            if sw > 0
                cen(k, :) = [sum(w .* byy), sum(w .* bxx)] / sw;
            end
        end

        va = ~isnan(rk);
        if sum(va) >= 2
            rv    = rk(va);
            cc    = cen(all(~isnan(cen), 2), :);
            drift = 0;
            for p1 = 1:size(cc, 1)
                for p2 = p1 + 1:size(cc, 1)
                    drift = max(drift, norm(cc(p1, :) - cc(p2, :)));
                end
            end
            feats(i, :) = [stab_global, mean(rv), min(rv), std(rv), ...
                           stab_global - min(rv), drift, sum(va) / K];
        end
    end

    feature_names = fnames; %#ok<NASGU>
    save(fullfile(session_dir, 'motion_diag.mat'), 'feats', 'feature_names', 'N');
    fprintf('[%s] motion_diag.mat saved (%d/%d candidates scored)\n', ...
            nm, sum(~isnan(feats(:, 1))), N);
end


function r = local_corr(img, abb_c, na)
% Pearson spatial correlation between an activity image and the (already
% mean-subtracted) footprint. Base MATLAB only -- no Stats toolbox.
    ic = img - mean(img);
    ni = sqrt(sum(ic .^ 2));
    if ni < 1e-9
        r = 0;
    else
        r = sum(ic .* abb_c) / (ni * na);
    end
end
