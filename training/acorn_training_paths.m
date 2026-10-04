function P = acorn_training_paths(trainee, root_override)
% acorn_training_paths -- machine-level folder for one trainee's progress.
%
%   P = acorn_training_paths(trainee)
%   P = acorn_training_paths(trainee, root_override)   (tests)
%
% Progress spans sessions, so it lives outside any session folder:
%   <userpath>/acorn_training/<trainee>/progress.csv
% userpath is per Windows user (prefdir is the fallback when it is empty), so
% two trainees sharing a machine under different logins keep separate files.
% The folder is created if missing (local disk only).

if nargin < 2 || isempty(root_override)
    up = userpath;
    if isempty(up); up = prefdir; end
    parts = strsplit(up, pathsep);
    up = parts{1};
    base = fullfile(up, 'acorn_training');
else
    base = root_override;
end
safe = regexprep(strtrim(char(trainee)), '[^A-Za-z0-9._-]+', '_');
safe = regexprep(safe, '^_+|_+$', '');
if isempty(safe); safe = 'trainee'; end
P = struct();
P.base = base;
P.trainee = safe;
P.root = fullfile(base, safe);
P.progress_csv = fullfile(P.root, 'progress.csv');
P.trainee_txt = fullfile(base, 'last_trainee.txt');
if ~exist(P.root, 'dir'); mkdir(P.root); end
end
