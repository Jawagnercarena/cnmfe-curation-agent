function S = acorn_training_score(decision, motion, key, opts)
% acorn_training_score -- compare a trainee's decisions with the answer key.
%
%   S = acorn_training_score(decision, motion, key, opts)
%
% decision : n x 1, 1 = keep, 0 = delete, NaN = never visited (counted as keep,
%            which is what viewNeurons does with an unvisited neuron)
% motion   : n x 1, 1 = the trainee pressed 'm' (motion delete)
% key      : from acorn_training_load_key
% opts.subset_mask       logical n x 1 (default all true): items in this attempt
% opts.exclude_contested (default true): items where the reference reviewer and
%                        the model disagree are shown but not counted
%
% Pure function, no I/O.  Scored set = subset & ~contested.  Over it:
%   a = both keep, b = trainee keep / reference delete (FALSE KEEP),
%   c = trainee delete / reference keep (FALSE DELETE), d = both delete
%   agreement = (a+d)/n, pe = ((a+b)(a+c) + (c+d)(b+d))/n^2,
%   kappa = (agreement - pe)/(1 - pe)   (NaN when pe == 1 or n == 0)
% outcome per item: 0 agree, 1 false keep, 2 false delete, 3 contested,
%                   4 outside the subset.
% Contested items are also summarised, uncounted: contested_with_ref /
% contested_with_model = how many of them you decided like the reference /
% like the model (key.model_keep).

if nargin < 4 || isempty(opts); opts = struct(); end
n = numel(key.ref_keep);
decision = double(decision(:));
motion = double(motion(:));
if numel(decision) ~= n || numel(motion) ~= n
    error('acorn_training_score: decision/motion must have n_review = %d entries', n);
end

unvisited = isnan(decision);
dec = decision;
dec(unvisited) = 1;
motion(isnan(motion)) = 0;
motion(unvisited) = 0;

subset = true(n, 1);
if isfield(opts, 'subset_mask') && ~isempty(opts.subset_mask)
    subset = logical(opts.subset_mask(:));
    if numel(subset) ~= n
        error('acorn_training_score: subset_mask must have %d entries', n);
    end
end
excl = true;
if isfield(opts, 'exclude_contested') && ~isempty(opts.exclude_contested)
    excl = logical(opts.exclude_contested);
end

con = logical(key.contested(:));
ref = logical(key.ref_keep(:));
tr = dec == 1;
scored = subset & ~(excl & con);

a = sum(scored & tr & ref);
b = sum(scored & tr & ~ref);
c = sum(scored & ~tr & ref);
d = sum(scored & ~tr & ~ref);
nsc = a + b + c + d;
if nsc > 0
    agreement = (a + d) / nsc;
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / nsc^2;
    if abs(1 - pe) < 1e-12
        kappa = NaN;
    else
        kappa = (agreement - pe) / (1 - pe);
    end
else
    agreement = NaN; pe = NaN; kappa = NaN;
end

outcome = zeros(n, 1);
outcome(~subset) = 4;
outcome(subset & excl & con) = 3;
outcome(scored & tr & ~ref) = 1;
outcome(scored & ~tr & ref) = 2;

csel = subset & con;
if isfield(key, 'model_keep') && ~isempty(key.model_keep)
    mk = double(key.model_keep(:)) == 1;
else
    mk = ~ref;                      % by definition the model disagrees on contested items
end
contested_with_ref = sum(csel & (tr == ref));
contested_with_model = sum(csel & (tr == mk));

refm = logical(key.ref_motion(:));
if key.has_motion_field == 1
    motion_ref_n = sum(scored & refm);
    motion_tagged_m = sum(scored & refm & motion == 1);
    motion_deleted_any = sum(scored & refm & ~tr);
    motion_false_tags = sum(scored & motion == 1 & ~refm);
else
    motion_ref_n = NaN; motion_tagged_m = NaN;
    motion_deleted_any = NaN; motion_false_tags = NaN;
end

codes = double(key.hint_code(:));
code_list = [0 1 2 3 4 5 6 7 9];
error_types = zeros(numel(code_list), 3);
for i = 1:numel(code_list)
    error_types(i, :) = [code_list(i), ...
        sum(outcome == 1 & codes == code_list(i)), ...
        sum(outcome == 2 & codes == code_list(i))];
end

rc = double(key.review_col(:));
S = struct();
S.n = n;
S.n_subset = sum(subset);
S.n_scored = nsc;
S.n_contested = sum(subset & con);
S.contested_with_ref = contested_with_ref;
S.contested_with_model = contested_with_model;
S.n_unvisited = sum(unvisited & subset);
S.n_ref_keep_scored = a + c;
S.n_ref_delete_scored = b + d;
S.a = a; S.b = b; S.c = c; S.d = d;
S.agreement = agreement;
S.pe = pe;
S.kappa = kappa;
S.false_keep = b;
S.false_delete = c;
S.motion_ref_n = motion_ref_n;
S.motion_tagged_m = motion_tagged_m;
S.motion_deleted_any = motion_deleted_any;
S.motion_false_tags = motion_false_tags;
S.outcome = outcome;
S.decision_used = dec;
S.false_keep_cols = rc(outcome == 1);
S.false_delete_cols = rc(outcome == 2);
S.contested_cols = rc(subset & con);
S.unvisited_cols = rc(unvisited & subset);
S.error_types = error_types;      % [hint_code, n_false_keep, n_false_delete]
S.exclude_contested = excl;
end
