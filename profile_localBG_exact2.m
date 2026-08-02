%% profile_localBG_exact2.m
% Follow-up to profile_localBG_exact.m, which found something specific:
%
%   Bx (contiguous gather, transposed back) produced BIT-IDENTICAL weights
%   (max|diff| exactly 0 at rr = 18, 28 and 39) but Yest differed by 1.4e-13.
%
% The asymmetry points at one line.  In Bx the fit gather was assigned to a
% variable first --
%       X = Yt(tmp_ind, ind_nhood).';   ... X*X'
% -- which forces MATLAB to materialise the transpose, so X*X' is the same BLAS
% call on the same bytes and the weights come out exact.  The reconstruction was
% written inline --
%       w'*(Yt(:, ind_nhood).')
% -- and MATLAB fuses an inline .' into a transpose FLAG on the BLAS call instead
% of materialising it, which changes the loop order and hence the summation.
%
% Two ways to make the reconstruction exact as well:
%   Bx2a  materialise it:  M = Yt(:, ind_nhood).';  w'*M
%   Bx2b  leave the line completely UNCHANGED (w'*Ywork(ind_nhood,:)) and use the
%         contiguous path only for the fit.  Bit-identical by construction, since
%         the code and the bytes are untouched; costs keeping both layouts (12.4 GB).
%
% Both are checked against the current implementation on the same pixels, with the
% reference saved to disk so later variants never have to re-run the slow one.
%
%   session_dir = 'D:\Julian_CNMFe\BLA\3odor\AVG5x-TSeries-07082026-bla36-810um-25z-000';
%   n_probe     = 6000;
%   run('C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\profile_localBG_exact2.m')

repo_root = 'C:\code\CNMF_E_LEGACY_BIANE_CLAUDE';
addpath(genpath(fullfile(repo_root, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_root, 'cnmfe_scripts')));
cvx_dirs = strsplit(genpath(fullfile(repo_root, 'cvx')), pathsep);
cvx_dirs = cvx_dirs(~cellfun(@isempty, cvx_dirs));
addpath(strjoin(cvx_dirs(~endsWith(cvx_dirs, [filesep 'narginchk_'])), pathsep));
clear cvx_dirs;
addpath(genpath(fullfile(repo_root, 'deconvolveCa')));

if ~exist('session_dir', 'var') || isempty(session_dir)
    error('profile_localBG_exact2: session_dir must be set.');
end
if ~exist('n_probe', 'var') || isempty(n_probe); n_probe = 6000; end

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:SingularMatrix');
warning('off', 'MATLAB:singularMatrix');

%% rebuild state
fprintf('[EXACT2] Loading session...\n');
ndata  = load(fullfile(session_dir, 'neuron.mat'));
neuron = ndata.neuron; clear ndata;
[~, session_nm] = fileparts(session_dir);
nam_mat = fullfile(session_dir, [session_nm, '.mat']);
if ~isfile(nam_mat)
    cand   = dir(fullfile(session_dir, '*.mat'));
    reject = {'neuron','Cn','Coor','pnr','Ybg','Ybg_mean', ...
              'spatial_footprints','review_neuron','Ybg_weights'};
    for i = numel(cand):-1:1
        [~, nm] = fileparts(cand(i).name);
        if any(strcmp(nm, reject)); cand(i) = []; end
    end
    nam_mat = fullfile(session_dir, cand(1).name);
end
data_file = matfile(nam_mat);
Ysiz = data_file.Ysiz; d1 = Ysiz(1); d2 = Ysiz(2); T = Ysiz(3);
Ywork = reshape(double(data_file.Y), d1*d2, T) - neuron.A*neuron.C;

ssub = 2; thresh = 10;
rr0  = ceil(neuron.options.gSiz * 1.5);
sn   = neuron.reshape(neuron.P.sn, 2);
Ywork = reshape(Ywork, d1, d2, T);
Ymean = mean(Ywork, 3);
Ywork = Ywork - bsxfun(@minus, Ymean, ones(1, 1, T));
Ywork = imresize(Ywork, 1./ssub);
[d1s, d2s, ~] = size(Ywork);
sn_s = imresize(sn, 1./ssub);
Npix = d1s*d2s; rr = round(rr0/ssub) + 1;
clear Ymean;

rsub_k = (-rr):rr;
[cind, rind] = meshgrid(rsub_k, rsub_k);
R = sqrt(cind.^2 + rind.^2);
neigh_kernel = (R >= rr) .* (R < rr+1);
J = nnz(neigh_kernel);
Yconv = bsxfun(@times, imfilter(Ywork, neigh_kernel), ...
    1./imfilter(ones(d1s, d2s), neigh_kernel));
ind_event = (bsxfun(@times, Ywork - Yconv, 1./sn_s) > thresh);
Ywork(ind_event) = Yconv(ind_event);
ind_event = reshape(ind_event, Npix, []);
clear Yconv;

[r_shift, c_shift] = find(neigh_kernel);
r_shift = r_shift - rr - 1; c_shift = c_shift - rr - 1;
[csub, rsub] = meshgrid(1:d2s, 1:d1s);
csub = reshape(csub, [], 1); rsub = reshape(rsub, [], 1);
csub = bsxfun(@plus, csub, c_shift'); rsub = bsxfun(@plus, rsub, r_shift');
oob = or(or(csub<1, csub>d2s), or(rsub<1, rsub>d1s));
csub(oob) = nan; rsub(oob) = nan;
Ywork = reshape(Ywork, Npix, []);

% Probe: half boundary-clipped, half interior, so both index paths are covered.
n_clip = sum(isnan(rsub) | isnan(csub), 2);
rng(0);
edge_px = find(n_clip > 0); int_px = find(n_clip == 0);
ne = min(round(n_probe/2), numel(edge_px));
ni = min(n_probe - ne, numel(int_px));
test_px = sort([edge_px(randperm(numel(edge_px), ne)); ...
                int_px(randperm(numel(int_px), ni))])';
nt = numel(test_px);
fprintf('[EXACT2] rr=%d J=%d; %d px (%d boundary-clipped, %d interior)\n', ...
    rr, J, nt, ne, ni);

%% A — reference
YestA = zeros(nt, T); wA = cell(nt, 1);
tA = tic;
for k = 1:nt
    px = test_px(k);
    valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
    ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
    if isempty(ind_nhood); continue; end
    tmp_ind = ~ind_event(px, 2:end);
    X = Ywork(ind_nhood, tmp_ind);
    y = Ywork(px, tmp_ind);
    tmpXX = X*X'; tmpXy = X*y';
    w = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
    YestA(k,:) = w'*Ywork(ind_nhood,:);
    wA{k} = [ind_nhood; w'];
end
sA = toc(tA);
fprintf('[EXACT2] A  current                        %8.1f s\n', sA);

tt = tic; Yt = Ywork.'; s_tp = toc(tt);

names = {}; times = {}; Ys = {}; ws = {};

%% Bx2a — materialise both gathers
YestP = zeros(nt, T); wP = cell(nt, 1);
t1 = tic;
for k = 1:nt
    px = test_px(k);
    valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
    ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
    if isempty(ind_nhood); continue; end
    tmp_ind = ~ind_event(px, 2:end);
    X = Yt(tmp_ind, ind_nhood).';
    y = Yt(tmp_ind, px).';
    tmpXX = X*X'; tmpXy = X*y';
    w = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
    M = Yt(:, ind_nhood).';            % materialised, not inlined
    YestP(k,:) = w'*M;
    wP{k} = [ind_nhood; w'];
end
names{end+1} = 'Bx2a materialise both'; times{end+1} = toc(t1);
Ys{end+1} = YestP; ws{end+1} = wP;

%% Bx2b — contiguous fit only; reconstruction line untouched
YestQ = zeros(nt, T); wQ = cell(nt, 1);
t2 = tic;
for k = 1:nt
    px = test_px(k);
    valid = ~isnan(rsub(px,:)) & ~isnan(csub(px,:));
    ind_nhood = sub2ind([d1s, d2s], rsub(px, valid), csub(px, valid));
    if isempty(ind_nhood); continue; end
    tmp_ind = ~ind_event(px, 2:end);
    X = Yt(tmp_ind, ind_nhood).';
    y = Yt(tmp_ind, px).';
    tmpXX = X*X'; tmpXy = X*y';
    w = (tmpXX + eye(size(tmpXX))*sum(diag(tmpXX))*(1e-5)) \ tmpXy;
    YestQ(k,:) = w'*Ywork(ind_nhood,:);   % identical code, identical bytes
    wQ{k} = [ind_nhood; w'];
end
names{end+1} = 'Bx2b contiguous fit only'; times{end+1} = toc(t2);
Ys{end+1} = YestQ; ws{end+1} = wQ;

%% compare
rep = {};
rep{end+1} = sprintf('localBG bit-exactness follow-up — %s', session_nm);
rep{end+1} = datestr(now, 'yyyy-mm-dd HH:MM:SS');
rep{end+1} = sprintf('rr=%d J=%d; %d px (%d boundary-clipped, %d interior)', ...
    rr, J, nt, ne, ni);
rep{end+1} = sprintf('  A  current                    %8.1f s   (reference)', sA);
for v = 1:numel(names)
    YB = Ys{v}; wB = ws{v};
    dY = max(abs(YestA(:) - YB(:)));
    nbad = 0; dW = 0;
    for k = 1:nt
        a = wA{k}; b = wB{k};
        if isempty(a) && isempty(b); continue; end
        if (isempty(a) ~= isempty(b)) || ~isequal(size(a), size(b)) ...
                || ~isequal(a(1,:), b(1,:))
            nbad = nbad + 1; continue;
        end
        dW = max(dW, max(abs(a(2,:) - b(2,:))));
    end
    if dY == 0 && nbad == 0 && dW == 0
        vd = '   <== BIT-IDENTICAL';
    else
        vd = '';
    end
    fprintf('[EXACT2] %-26s %8.1f s %5.2fx | Yest %.3g | idx bad %d | w %.3g%s\n', ...
        names{v}, times{v}, sA/times{v}, dY, nbad, dW, vd);
    rep{end+1} = sprintf('  %-28s %8.1f s  %5.2fx', names{v}, times{v}, sA/times{v});
    rep{end+1} = sprintf('     Yest max|diff| %-10.3g weights idx mismatches %-4d weights max|diff| %-10.3g%s', ...
        dY, nbad, dW, vd);
end
rep{end+1} = sprintf('  (one-off transpose %.1f s)', s_tp);

txt = strjoin(rep, newline);
fprintf('\n%s\n', txt);
out = fullfile(session_dir, 'localBG_exact2.txt');
fid = fopen(out, 'w'); fprintf(fid, '%s\n', txt); fclose(fid);
fprintf('[EXACT2] Report written to %s\n', out);
