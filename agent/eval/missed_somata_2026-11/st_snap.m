function st_snap(nrn, tag, mROI, nIDs)
% SEEDTRACE snapshot of the component set (agent/eval/missed_somata_2026-11).
% Reads nrn.A only; never modifies the Sources2D object. Each footprint is
% stored as (pixel, component, value/max) triplets above 0.1 of its own max.
% mROI / nIDs are the merged_ROI / newIDs returned by the merge step just
% run (pre-merge indices), or [] for steps that do not merge.
global ST_SNAPS
A = nrn.A;
K = size(A, 2);
s = struct();
s.tag = tag;
s.K = K;
s.npix = size(A, 1);
if K > 0
    mx = full(max(A, [], 1));
    [i, j, v] = find(A);
    rel = full(v) ./ reshape(mx(j), [], 1);
    keep = rel >= 0.1;
    s.i = uint32(i(keep));
    s.j = uint16(j(keep));
    s.v = single(rel(keep));
    s.mx = single(mx);
else
    s.i = uint32([]); s.j = uint16([]); s.v = single([]); s.mx = single([]);
end
if isempty(mROI)
    s.merged = {};
else
    s.merged = mROI;
end
s.newIDs = nIDs;
if isempty(ST_SNAPS); ST_SNAPS = {s}; else; ST_SNAPS{end+1} = s; end
fprintf('[SEEDTRACE] snap %-10s K=%d merges=%d\n', tag, K, numel(s.merged));
end
