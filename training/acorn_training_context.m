function ctx = acorn_training_context(neuron, Cn, key)
% acorn_training_context -- per-session display state for the training tools.
%
%   ctx = acorn_training_context(neuron, Cn, key)
%
% Computed once from the UNMUTATED Sources2D object in review_neuron.mat:
%   perm        display order = what the real review shows after
%               neuron.orderROIs('mean'): [~, perm] = sort(mean(C, 2), 'descend')
%               (Sources2D.m, 'mean' branch; order_ROIs applies it as a pure
%               permutation).  The object itself is never reordered here, so
%               review column k of the key stays column k.
%   pos_of_col  inverse permutation (shown "Neuron N" number of each column)
%   Amask, ctr, gSiz, Amax, t, str_xlabel   exactly what viewNeurons.m uses
%   Cn_full     the correlation image resized to d1 x d2 (CNMFe_final_save does
%               the same before plotting contours)
%   contours    get_contours(0.8) for every column (image coordinates)
% Errors if the candidate count differs from the key's n_review.

if nargin < 3; key = []; end
n = size(neuron.A, 2);
if ~isempty(key) && n ~= key.n_review
    error(['review_neuron.mat has %d candidates but training_key.mat expects %d. ' ...
           'The key was built for a different candidate set -- ask the operator ' ...
           'to rebuild it.'], n, key.n_review);
end
ctx = struct();
ctx.n = n;
ctx.d1 = neuron.options.d1;
ctx.d2 = neuron.options.d2;
if isfield(neuron.options, 'ssub') && ~isempty(neuron.options.ssub)
    ctx.ssub = neuron.options.ssub;
else
    ctx.ssub = 1;
end
ctx.gSiz = neuron.options.gSiz;
ctx.Amask = (neuron.A > 0);
ctx.ctr = neuron.estCenter();
ctx.Amax = full(max(neuron.A, [], 1));
T = size(neuron.C, 2);
t = 1:T;
if ~isnan(neuron.Fs) && neuron.Fs > 0
    t = t / neuron.Fs;
    ctx.str_xlabel = 'Time (Sec.)';
else
    ctx.str_xlabel = 'Frame';
end
ctx.t = t;
[~, perm] = sort(full(mean(neuron.C, 2)), 'descend');
ctx.perm = perm(:);
ctx.pos_of_col = zeros(n, 1);
ctx.pos_of_col(ctx.perm) = (1:n)';
if isempty(Cn)
    ctx.Cn_full = [];
else
    Cn = double(Cn);
    if ~isequal(size(Cn), [ctx.d1, ctx.d2])
        Cn = imresize(Cn, [ctx.d1, ctx.d2]);
    end
    ctx.Cn_full = Cn;
end
ctx.contours = cell(n, 1);
try
    ctx.contours = neuron.get_contours(0.8);
catch err
    fprintf(2, 'acorn_training_context: contours unavailable (%s); Cn panel shows no outline.\n', err.message);
end
end
