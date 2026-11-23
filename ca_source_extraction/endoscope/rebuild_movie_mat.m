function mat_path = rebuild_movie_mat(tif_path, mat_path)
%REBUILD_MOVIE_MAT  Rebuild a session's CNMF-E movie .mat (Y, Ysiz) from its .tif.
%
%   mat_path = rebuild_movie_mat(tif_path, mat_path)
%
%   Writes the same content as tif2mat.m (same frame count rule, same reader
%   smod_bigread2, same variables Y and Ysiz, same -v7.3 format) with two
%   differences:
%     * the output path is given instead of derived from the tif name;
%     * the file is written under <folder of mat_path>/.movie_rebuild/ first,
%       checked, and only then moved into place, so an interrupted rebuild never
%       leaves a truncated movie under the real name (cnmfe_choose_data and
%       CNMFe_final_save accept any existing movie .mat as finished).
%
%   Used once a finished session's movie has been pruned
%   (agent/prune_movie_mats.py): CNMFe_final_save.m and the bundle pushes
%   (agent/movie_mat.py) call it.  Never overwrites an existing mat_path.

if isfile(mat_path)
    error('rebuild_movie_mat:exists', 'Refusing to overwrite the existing movie %s', mat_path);
end
if ~isfile(tif_path)
    error('rebuild_movie_mat:noTif', 'Source tif not found: %s', tif_path);
end
[out_dir, out_nm, out_ext] = fileparts(mat_path);
tmp_dir  = fullfile(out_dir, '.movie_rebuild');
tmp_path = fullfile(tmp_dir, [out_nm, out_ext]);
if ~isfolder(tmp_dir)
    mkdir(tmp_dir);
end
t_start = tic;

% ---- frame count and size: the same rule as tif2mat.m ----
info = imfinfo(tif_path);
if isfield(info(1), 'ImageDescription') && ~isempty(strfind(info(1).ImageDescription, 'ImageJ'))
    junk1 = regexp(info(1).ImageDescription, 'images=\d*', 'match');
    if isempty(junk1)
        junk1 = regexp(info(1).ImageDescription, 'frames=\d*', 'match');
    end
    junk2 = strjoin(junk1);
    T = strread(junk2, '%*s %d', 'delimiter', '='); %#ok<DSTRRD>
else
    T = length(info);   % number of frames
end
d1 = info(1).Height;
d2 = info(1).Width;
Ysiz = [d1, d2, T]';
clear info;
fprintf('Rebuilding %s from %s (%d x %d x %d frames) ...\n', [out_nm, out_ext], tif_path, d1, d2, T);

% ---- write: the same chunking as tif2mat.m, into the temp folder ----
Tchunk = min(T, round(2^29/d1/d2));
Y = smod_bigread2(tif_path, 1, Tchunk);
save(tmp_path, 'Y', 'Ysiz', '-v7.3');
clear Y;
if Tchunk < T
    data = matfile(tmp_path, 'Writable', true);
    t0 = Tchunk + 1;
    while t0 <= T
        num2read = min(t0 + Tchunk - 1, T) - t0 + 1;
        tmpY = smod_bigread2(tif_path, t0, num2read);
        data.Y(:, :, (1:num2read) + t0 - 1) = tmpY;
        t0 = t0 + num2read;
    end
    clear data tmpY;
end

% ---- check before moving into place: size, Ysiz, first/middle/last frame ----
chk = matfile(tmp_path);
sz = size(chk, 'Y');
sz(end+1:3) = 1;
ysz = chk.Ysiz;          % matfile objects do not support linear indexing
ysz = double(ysz(:)');
if ~isequal(double(sz(1:3)), double([d1, d2, T])) || ~isequal(ysz, double([d1, d2, T]))
    error('rebuild_movie_mat:check', ...
          'Rebuilt movie has size %s / Ysiz %s, expected %d x %d x %d; left at %s', ...
          mat2str(sz), mat2str(ysz), d1, d2, T, tmp_path);
end
for t = unique([1, round(T/2), T])
    if ~isequal(chk.Y(:, :, t), smod_bigread2(tif_path, t, 1))
        error('rebuild_movie_mat:check', ...
              'Frame %d of the rebuilt movie does not match the tif; left at %s', t, tmp_path);
    end
end
clear chk;

% ---- move into place (same folder tree, so this is a rename) ----
if isfile(mat_path)
    error('rebuild_movie_mat:race', ...
          '%s appeared while rebuilding; the new copy was left at %s', mat_path, tmp_path);
end
[ok, msg] = movefile(tmp_path, mat_path);
if ~ok
    error('rebuild_movie_mat:move', 'Could not move %s to %s: %s', tmp_path, mat_path, msg);
end
[~, ~] = rmdir(tmp_dir);   % only removes the folder when it is empty
fprintf('Rebuilt %s in %.0f s.\n', mat_path, toc(t_start));
end
