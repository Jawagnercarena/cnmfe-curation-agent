%% clean_matlab_path.m
% One-time maintenance: strips dead entries out of MATLAB's *saved* search path
% (pathdef.m), which is what makes every headless launch open with
%
%   Warning: Name is nonexistent or not a directory: D:\Julian_CNMFe\CTA
%   Warning: Function narginchk has the same name as a MATLAB built-in...
%
% Those fire while MATLAB loads pathdef.m at startup, before any agent code
% runs, so they can only be fixed here. Two kinds of entry are cleaned:
%
%   - Directories that no longer exist, usually data folders the CNMF-E GUI put
%     on the path and someone then saved. MATLAB warns about each one at
%     startup and then drops it, so they are already absent from `path`;
%     savepath rewrites pathdef.m from the live path, which is what removes
%     them for good.
%   - cvx's lib/narginchk_ shim, which cvx_startup adds only on pre-R2013a
%     MATLAB but genpath() adds unconditionally. It shadows the builtin.
%
% Run this from a normal interactive MATLAB session. pathdef.m sits inside the
% MATLAB install directory; savepath can usually still write it, but if it
% cannot, restart MATLAB with "Run as administrator" and run this again.
% savepath only rewrites the user portion of pathdef.m - the MathWorks toolbox
% entries come from a template and are left alone.
%
% Removal-only and safe to re-run.

live = strsplit(path, pathsep);
live = live(~cellfun(@isempty, live));

gone  = live(cellfun(@(d) exist(d, 'dir') ~= 7, live));
shims = live(endsWith(live, [filesep 'narginchk_']));
stale = unique([gone, shims]);

if ~isempty(stale)
    fprintf('Removing %d stale entry/entries from the path:\n', numel(stale));
    fprintf('    %s\n', stale{:});
    rmpath(stale{:});
    fprintf('\n');
else
    fprintf('Nothing stale on the active path.\n');
end

if savepath == 0
    fprintf(['Saved to %s\n' ...
             'Restart MATLAB - any startup path warnings should be gone.\n'], ...
            which('pathdef.m'));
else
    fprintf(2, ['savepath FAILED: %s is not writable.\n' ...
                'Close MATLAB, restart it as administrator (right-click the ' ...
                'MATLAB shortcut -> Run as administrator), and run this ' ...
                'script again.\n'], which('pathdef.m'));
end
