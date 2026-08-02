function [Yest, results] = local_background(Y, ssub, rr, ACTIVE_PX, sn, thresh, p_cutoff)
%% approximate the background with locally-linear embedding
%% inputs:
%   Y: d1*d2*T 3D matrix, video data
%   rr: scalar, average neuron size
%   ssub: spatial downsampling factor
%   ACTIVE_PX:  indicators of pixels to be approximated
%   sn:  noise level for each pixel 
%   thresh: threshold for selecting frames with resting states 
%   p_cutoff: pick neighbors whose corr. coefficients with the center pixel 
%   are smaller than quantiles of p_cutoff. 
%% outputs:
%   Yest: d1*d2*T 3D matrix, reconstructed video data
%   results: struct variable {weights, ssub}
%       weights: d1*d2 cell, each element is a 2*J matrix. Row 1 has the indice of the
%       ring neighbors and row 2 has the corresponding weights.
%       ssub:    scalar, spatial downsampling factor

%% Author: Pengcheng Zhou, Carnegie Mellon University,2016

%% input arguments
[d1, d2, T] = size(Y);

% center the fluorescence intensity by its mean
Ymean = mean(Y, 3);
Y = Y - bsxfun(@minus, Ymean, ones(1, 1, T));

% average neuron size
if ~exist('rr', 'var')|| isempty(rr)
    rr = 15;
end
% spatial downsampling
if ~exist('ssub', 'var') || isempty(ssub)
    ssub = 1;
end

%downsample the data
if ssub>1
    Y = imresize(Y, 1./ssub);
    [d1s, d2s, ~] = size(Y);
    rr = round(rr/ssub)+1;
    
    if ~exist('sn', 'var')||isempty(sn)
        sn = reshape(get_noise_fft(reshape(Y, d1s*d2s, [])), d1s, d2s);
    else
        sn = imresize(sn, 1./ssub);
    end
else
    d1s = d1;
    d2s = d2;
    if ~exist('sn', 'var')||isempty(sn)
        sn = reshape(get_noise_fft(reshape(Y, d1s*d2s, [])), d1s, d2s);
    end
end

% threshold for selecting frames with resting state 
if ~exist('thresh', 'var') || isempty(thresh) 
    thresh = 3; 
end

if ~exist('p_cutoff', 'var') || isempty(p_cutoff)
    p_cutoff = 1; 
end 

%% threshold the data
rsub = (-rr):(rr);      % row subscript
csub = rsub;      % column subscript
[cind, rind] = meshgrid(csub, rsub);
R = sqrt(cind.^2+rind.^2);
neigh_kernel = (R>=rr) .* (R<rr+1);  % kernel representing the selected neighbors 
Yconv = bsxfun(@times, imfilter(Y, neigh_kernel), ...
    1./imfilter(ones(d1s, d2s), neigh_kernel));
ind_event = (bsxfun(@times, Y-Yconv, 1./sn)> thresh); % frames with larger signal
Y(ind_event) = Yconv(ind_event); % remove potential calcium transients
ind_event = reshape(ind_event, d1s*d2s, []);
clear Yconv;   % dead from here, and it is the same size as Y (6.2 GB at full res)

% pixels to be approximated
if exist('ACTIVE_PX', 'var') && ~isempty(ACTIVE_PX)
    ACTIVE_PX = reshape(double(ACTIVE_PX), d1, d2);
    ACTIVE_PX = (imresize(ACTIVE_PX, 1/ssub)>0);
    ind_px = find(ACTIVE_PX(:));  % pixels to be approxiamted
else
    ind_px = (1:(d1s*d2s));
end

%% determine neibours of each pixel
[r_shift, c_shift] = find(neigh_kernel);
r_shift = r_shift - rr -1;
c_shift = c_shift - rr - 1;

[csub, rsub] = meshgrid(1:d2s, 1:d1s);
csub = reshape(csub, [], 1);
rsub = reshape(rsub, [], 1);
csub = bsxfun(@plus, csub, c_shift');
rsub = bsxfun(@plus, rsub, r_shift');
% remove neighbors that are out of boundary
ind = or(or(csub<1, csub>d2s), or(rsub<1, rsub>d1s));
csub(ind) = nan;
rsub(ind) = nan;

%% run approximation
warning('off','MATLAB:nearlySingularMatrix'); 
warning('off','MATLAB:SingularMatrix'); 
% gamma = 0.001; % add regularization
Y = reshape(Y, d1s*d2s, []);
Yest = zeros(size(Y));
weights = cell(d1s, d2s);

% Y is stored pixel-major, so gathering a pixel's ring neighbours means reading
% J rows out of d1s*d2s — every element on its own cache line, which turned 17 MB
% of useful data into ~137 MB of memory traffic per pixel and left this loop
% running at a few percent of the machine's throughput.  Keeping a time-major copy
% makes each neighbour's samples contiguous instead, which is where the speedup
% comes from (measured 1.60x; the transpose itself costs 0.6 s and is offset by
% the Yconv clear above).
%
% Each gather below is assigned to a variable before use, and that is load-bearing,
% not stylistic: written inline, MATLAB folds a .' into a transpose FLAG on the
% BLAS call rather than materialising the array, which reorders the summation and
% shifts results by ~1e-12.  Assigning forces the array to be built, so X*X' is
% the same call on the same bytes as the pixel-major version.  Verified bit-exact
% over 1,041,051,648 values (all 65,536 pixels at rr=28, plus boundary-weighted
% samples at rr=18 and rr=39) — zero differing elements, weights included.
% See profile_localBG_confirm.m.
Yt = Y.';

for m=1:length(ind_px)
    px = ind_px(m);
    % Filter NaN neighbors (boundary-adjacent pixels) BEFORE sub2ind.
    % The original code set out-of-boundary neighbors to NaN (lines 97-99)
    % and intended to clean up after sub2ind, but sub2ind throws on NaN
    % input. This manifests when neuron footprints exist near image edges,
    % which happens in headless mode (no pre-BG manual deletion step).
    valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
    ind_nhood = sub2ind([d1s,d2s], rsub(px, valid), csub(px, valid));
    if isempty(ind_nhood)
        continue;  % corner pixel with no valid neighbors — skip
    end
%     J = length(ind_nhood);
    
    tmp_ind = ~ind_event(px, 2:end);
    X = Yt(tmp_ind, ind_nhood).';   % contiguous read; .' assigned, never inlined
    y = Yt(tmp_ind, px).';
    tmpXX = X*X';
    tmpXy = X*y';
    if p_cutoff<1
        temp = tmpXy./diag(tmpXX); 
        idx = (temp < quantile(temp, p_cutoff)); 
        tmpXX = tmpXX(idx, idx); 
        tmpXy = tmpXy(idx); 
        ind_nhood = ind_nhood(idx); 
    end 
    w = (tmpXX+eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
    M = Yt(:, ind_nhood).';         % likewise assigned, so w'*M is the same call
    Yest(px, :) = w'*M;
    weights{px} = [ind_nhood; w'];
end
results.weights = weights;
results.ssub = ssub;
results.dims = [d1s, d2s]; 

ind = 1:(d1s*d2s);
ind(ind_px) = [];
if ~isempty(ind)
    temp = imfilter(Y, ones(3, 3)/9, 'replicate');
    Yest(ind, :) = temp(ind, :); % without approximation
end
Yest = reshape(Yest, d1s, d2s, []);
warning('on','MATLAB:nearlySingularMatrix'); 
warning('on','MATLAB:SingularMatrix'); 
%% return the result
if ssub>1 %up sampling
    Yest = imresize(Yest, [d1, d2]);
end
%
% clear Y;
Ybaseline = Ymean-mean(Yest,3); % medfilt2(Ymean, [3,3]); % - median(Yest, 3);
Yest = bsxfun(@plus, Yest, Ybaseline);
results.b0 = Ybaseline; 
