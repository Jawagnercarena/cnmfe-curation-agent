function [peaks, onsets] = find_trace_transients(c, k, minsep, onset_frac)
% find_trace_transients -- the biggest transients of one calcium trace.
%
%   [peaks, onsets] = find_trace_transients(c, k, minsep, onset_frac)
%
% c          1 x T trace (the fitted calcium trace neuron.C works best)
% k          at most this many transients, highest peak first (default 5)
% minsep     peaks closer than this many frames to a higher one are skipped
%            (default 10)
% onset_frac onset = the last frame before the peak where the trace is still
%            rising and above onset_frac * peak (default 0.1); walking back
%            from the peak stops at a dip or once the trace drops under that
%            fraction, so for a clean transient it is the first rising frame
%
% Pure function, no toolboxes.  Used by viewNeuronsVideo (keys n / p) and by
% the training drill so a jump lands BEFORE the transient starts and playback
% shows the rise, which is what separates a cell firing in place from the
% frame moving.  A flat trace returns its maximum as a single pseudo peak.

if nargin < 2 || isempty(k); k = 5; end
if nargin < 3 || isempty(minsep); minsep = 10; end
if nargin < 4 || isempty(onset_frac); onset_frac = 0.1; end
c = double(full(c(:)))';
n = numel(c);
if n < 3
    [~, peaks] = max(c);
    onsets = peaks;
    return;
end
cand = find(c(2:end-1) > c(1:end-2) & c(2:end-1) >= c(3:end)) + 1;
if isempty(cand)
    [~, cand] = max(c);
end
[~, order] = sort(c(cand), 'descend');
cand = cand(order);
peaks = [];
for i = 1:numel(cand)
    if isempty(peaks) || all(abs(cand(i) - peaks) >= minsep)
        peaks(end+1) = cand(i); %#ok<AGROW>
    end
    if numel(peaks) >= k; break; end
end
onsets = zeros(size(peaks));
for j = 1:numel(peaks)
    pk = peaks(j);
    thr = onset_frac * c(pk);
    i = pk;
    while i > 1 && c(i-1) > thr && c(i-1) < c(i)
        i = i - 1;
    end
    onsets(j) = i;
end
end
