function motion_qc(session_dir, repo_root)
% Motion QC flag (advisory only -- deletes nothing, touches no model).
%
% TWO-CHANNEL flag. Motion artifacts come in different flavors and no single
% signal catches all of them, so we score two independent axes and flag a
% candidate that is suspect on EITHER:
%   lateral   = within-session z(mot_mean) + z(mot_p90)   -- local displacement
%               level (catches x-y drift, e.g. a neighbour sliding into the ROI)
%   structure = within-session z(s_z)                     -- onset-locked
%               structural change (catches axial z / plane-changes, which are a
%               brightness/structure change, NOT a sideways shift)
% Which axis dominates flips by session, and we cannot know which at flag time,
% so we do not collapse them -- we show both. A cell in the top quartile of a
% channel is flagged for extra scrutiny in the video pass. On the annotated
% image: lateral-suspect = red, structure-suspect = orange, both = yellow.
%
% Rationale: motion deletes are a large fraction of all deletes, and the model
% cannot catch them (they look like real cells in the extracted features), so
% its garbage-rejection is capped -- this points the human's eye at the cells
% the model must miss. It deletes nothing and touches no model / feature
% contract; the exact channels are validated on only ~8 sessions / 3-4 animals,
% so treat it as a soft aid and re-validate (test_motion_onset.py) as (m) labels
% accrue across more animals.
%
% Writes into session_dir:
%   motion_qc.mat  per-candidate severities + flags (lateral / structure / any)
%   motion_qc.jpg  candidate footprints tinted by channel
% Prints a ranked report; if labels.mat exists, self-validates against the (m)
% tags for each channel and the union.

    if ~exist('repo_root', 'var') || isempty(repo_root)
        repo_root = fileparts(fileparts(mfilename('fullpath')));  % repo = parent of agent/
    end
    addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));

    FLAG_PCTILE = 75;   % flag the top quartile of EACH channel (liberal by design)

    % --- Ensure motion_vec.mat exists AND carries the onset-structure channel ---
    vecfile = fullfile(session_dir, 'motion_vec.mat');
    recompute = true;
    if isfile(vecfile)
        mvd = load(vecfile);
        if isfield(mvd, 'feature_names') && any(strcmp(mvd.feature_names, 's_z'))
            recompute = false;
        end
    end
    if recompute
        fprintf('motion_vec.mat missing/outdated -- computing it (loads the movie)...\n');
        extract_motion_vectors(session_dir, repo_root);
        mvd = load(vecfile);
    end
    feats = mvd.feats;
    names = mvd.feature_names;
    colof = @(nm) find(strcmp(names, nm), 1);
    m_mean = feats(:, colof('mot_mean'));
    m_p90  = feats(:, colof('mot_p90'));
    s_z    = feats(:, colof('s_z'));
    N      = size(feats, 1);

    % --- Two within-session severity channels ---
    z = @(v) (v - mean(v, 'omitnan')) ./ (std(v, 'omitnan') + 1e-9);
    sev_lat = z(m_mean) + z(m_p90);                 % lateral displacement level
    sev_str = z(s_z);                               % onset-locked structure change
    [pct_lat, flag_lat] = pct_and_flag(sev_lat, FLAG_PCTILE);
    [pct_str, flag_str] = pct_and_flag(sev_str, FLAG_PCTILE);
    flag_any = flag_lat | flag_str;

    % --- Save the per-candidate QC ---
    save(fullfile(session_dir, 'motion_qc.mat'), ...
         'pct_lat', 'pct_str', 'flag_lat', 'flag_str', 'flag_any', 'FLAG_PCTILE');

    % --- Report ---
    [~, nm] = fileparts(session_dir);
    fprintf('\n=== Motion QC: %s ===\n', nm);
    fprintf(['%d candidates. Flagged: %d lateral (red), %d structure (orange), ' ...
             '%d both (yellow) -- %d total (top %d%% of each channel).\n'], ...
            N, sum(flag_lat & ~flag_str), sum(flag_str & ~flag_lat), ...
            sum(flag_lat & flag_str), sum(flag_any), 100 - FLAG_PCTILE);
    report_top('Most LATERAL-suspect (x-y drift)',   pct_lat, 8);
    report_top('Most STRUCTURE-suspect (axial z)',   pct_str, 8);

    % --- Self-validation against (m) tags, if labels exist ---
    lf = fullfile(session_dir, 'labels.mat');
    if isfile(lf)
        L = load(lf);
        if isfield(L, 'motion_delete')
            ym = logical(L.motion_delete(:));
            if numel(ym) == N && sum(ym) > 0
                nM = sum(ym); base = nM / N;
                fprintf('Self-check vs (m) tags: %d motion cells (%.0f%% of candidates).\n', ...
                        nM, 100 * base);
                fprintf('  channel     flag%%  recall  precision   enrichment\n');
                report_channel('lateral  ', flag_lat, ym, base);
                report_channel('structure', flag_str, ym, base);
                report_channel('either   ', flag_any, ym, base);
            end
        end
    end

    % --- Annotated footprint image (alignment-safe: footprint masks directly) ---
    try
        rn = load(fullfile(session_dir, 'review_neuron.mat'));
        A  = full(rn.neuron.A);
        d1 = double(rn.neuron.options.d1); d2 = double(rn.neuron.options.d2);
        bg = reshape(sum(A, 2), d1, d2); bg = bg ./ max(bg(:) + 1e-9);
        img = repmat(bg * 0.7, [1 1 3]);                 % grey footprint-sum
        R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
        for i = 1:N
            if ~flag_any(i); continue; end
            mask = reshape(A(:, i), d1, d2) > 0;
            if flag_lat(i) && flag_str(i)
                col = [1.0 0.9 0.1];        % both -> yellow
            elseif flag_lat(i)
                col = [1.0 0.1 0.1];        % lateral -> red
            else
                col = [1.0 0.6 0.1];        % structure -> orange
            end
            R(mask) = col(1); G(mask) = col(2); B(mask) = col(3);
        end
        img(:,:,1) = R; img(:,:,2) = G; img(:,:,3) = B;
        fig = figure('visible', 'off', 'position', [100 100 850 850]);
        imshow(img);
        title(sprintf(['Motion QC -- %s\nred=lateral(%d)  orange=structure(%d)  ' ...
                       'yellow=both(%d)'], nm, sum(flag_lat & ~flag_str), ...
                      sum(flag_str & ~flag_lat), sum(flag_lat & flag_str)), ...
              'Interpreter', 'none');
        saveas(fig, fullfile(session_dir, 'motion_qc.jpg')); close(fig);
        fprintf('Wrote motion_qc.mat and motion_qc.jpg\n');
    catch err
        fprintf(2, 'motion_qc.jpg skipped (%s); motion_qc.mat still written.\n', err.message);
    end
end


function [pct, flag] = pct_and_flag(sev, p)
% Within-session percentile rank (0-100, NaN preserved) + top-(100-p)% flag.
    N = numel(sev);
    pct = nan(N, 1);
    ok = ~isnan(sev);
    if any(ok)
        [~, ord] = sort(sev(ok));
        pr = zeros(sum(ok), 1); pr(ord) = (1:sum(ok))' / sum(ok) * 100;
        pct(ok) = pr;
    end
    flag = pct >= p;          % NaN percentile (unscored) -> not flagged
end


function report_top(header, pct, ntop)
    [sv, si] = sort(pct, 'descend', 'MissingPlacement', 'last');
    fprintf('%s:\n', header);
    for k = 1:min(ntop, sum(~isnan(pct)))
        fprintf('   #%-4d  severity %5.1f\n', si(k), sv(k));
    end
end


function report_channel(name, flag, ym, base)
    nf   = sum(flag);
    prec = sum(flag & ym) / max(nf, 1);
    rec  = sum(flag & ym) / sum(ym);
    fprintf('  %s  %4.0f%%   %4.0f%%      %4.0f%%       %.2fx\n', ...
            name, 100 * nf / numel(flag), 100 * rec, 100 * prec, prec / max(base, 1e-9));
end
