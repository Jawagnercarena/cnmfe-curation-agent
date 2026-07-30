function extract_motion_series(session_dir, repo_root)
% Export PER-FRAME local-motion and structure-change series for each review
% candidate, so the onset-coincidence analysis can be done in Python. This is a
% faithful extension of extract_motion_vectors.m: identical Y-load, bbox, and
% Lucas-Kanade math -- but instead of reducing each candidate to scalars, it
% saves the full time series so we can ask the frame-precise question ("does
% motion / structure-change spike right AT each transient onset?").
%
% Saved to motion_series.mat (per candidate row i, over all T frames):
%   mot(i,t)    per-frame |displacement| magnitude (coarse rigid LK vs resting)
%   sdec(i,t)   per-frame structural decorrelation, 1 - corr(patch(t), rest)
%   traces(i,t) the candidate's C_raw trace (for onset detection)
%   baseln(i)   trace median (baseline used for onset threshold)
%   sigma_(i)   robust trace sigma (MAD/0.6745)
%   scored(i)   true if the candidate was measurable (non-degenerate patch)
%
% Read-only except for writing motion_series.mat. ~1-2 min/session (movie load).

    addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));

    MARGIN  = 8;      % bbox margin (px) -- matches extract_motion_vectors.m
    K_ACT   = 2.5;    % activity threshold in robust sigmas (for the resting ref)
    MIN_ACT = 5;      % min active / inactive frames to score a candidate

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

    mot    = nan(N, T, 'single');
    sdec   = nan(N, T, 'single');
    traces = single(C);            % N x T, C_raw as-is
    baseln = nan(N, 1);
    sigma_ = nan(N, 1);
    scored = false(N, 1);

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
        bb  = sub2ind([d1 d2], Yg(:), Xg(:));       % pixel order = rows(y) fastest
        Ybb = double(Y(bb, :));                       % nbb x T

        ci = C(i, :);
        baseline = median(ci);
        sg       = median(abs(ci - baseline)) / 0.6745;
        if sg < 1e-9; continue; end
        thr    = baseline + K_ACT * sg;
        active = ci > thr;
        if sum(active) < MIN_ACT || sum(~active) < MIN_ACT; continue; end

        % Resting structure = mean over inactive (cell-silent) frames.
        ref     = mean(Ybb(:, ~active), 2);           % nbb x 1
        ref_img = reshape(ref, h, w);
        [gx, gy] = gradient(ref_img);
        G = [gx(:), gy(:)];                           % nbb x 2
        AtA = G' * G;
        if rcond(AtA) < 1e-6; continue; end           % flat/degenerate patch

        % Lucas-Kanade per-frame rigid displacement (vectorised over T).
        It  = Ybb - ref;                              % nbb x T
        D   = AtA \ (-(G' * It));                     % 2 x T -> [dx; dy]
        moti = sqrt(D(1, :) .^ 2 + D(2, :) .^ 2);     % 1 x T

        % Per-frame structural correlation to the resting patch.
        refc = ref - mean(ref); nref = sqrt(sum(refc .^ 2));
        Ybbc = Ybb - mean(Ybb, 1);
        pc   = (refc' * Ybbc) ./ (sqrt(sum(Ybbc .^ 2, 1)) * nref + 1e-9);  % 1 x T

        mot(i, :)  = single(moti);
        sdec(i, :) = single(1 - pc);
        baseln(i)  = baseline;
        sigma_(i)  = sg;
        scored(i)  = true;
    end

    feature_note = ['mot=LK |disp| per frame; sdec=1-corr(patch,rest) per frame; ' ...
                    'traces=C_raw; onset threshold = baseln + 2.5*sigma_']; %#ok<NASGU>
    % v7 (default) so scipy.io.loadmat can read it; arrays are ~tens of MB.
    save(fullfile(session_dir, 'motion_series.mat'), ...
         'mot', 'sdec', 'traces', 'baseln', 'sigma_', 'scored', 'N', 'T', ...
         'feature_note');
    fprintf('[%s] motion_series.mat saved (%d/%d scored)\n', nm, sum(scored), N);
end
