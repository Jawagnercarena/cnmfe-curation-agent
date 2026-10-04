function acorn_training_render_gallery(manifest_json)
% acorn_training_render_gallery -- render gallery example panels headless.
%
% Central-machine step launched by agent/build_training_gallery.py.  Reads the
% manifest (items with session_dir, review_col, category, partner_col, png,
% label), loads each session's review_neuron.mat once, draws every item with
% acorn_training_draw_panel (footprint, zoom, Cn + contour, trace) and prints
% <out_dir>/<png>.  Writes <out_dir>/rendered.json with, per item, png,
% shown_pos (the number the drill would show) and ok.  Session folders are
% only read.

txt = fileread(manifest_json);
M = jsondecode(txt);
out_dir = M.out_dir;
items = M.items;
if isstruct(items)
    items = num2cell(items);
end
n = numel(items);
fprintf('acorn_training_render_gallery: %d item(s) -> %s\n', n, out_dir);
img_dir = fullfile(out_dir, 'img');
if ~exist(img_dir, 'dir'); mkdir(img_dir); end

sessions = cellfun(@(it) char(it.session_dir), items, 'UniformOutput', false);
[usess, ~, gidx] = unique(sessions);
out = cell(n, 1);
for i = 1:n
    out{i} = struct('png', char(items{i}.png), 'shown_pos', 0, 'ok', false, 'note', '');
end
fig = figure('Visible', 'off', 'Position', [100, 100, 1280, 560]);
for s = 1:numel(usess)
    sd = usess{s};
    idx = find(gidx == s)';
    try
        rn = load(fullfile(sd, 'review_neuron.mat'));
        neuron = rn.neuron;
        Cn = []; if isfield(rn, 'Cn'); Cn = rn.Cn; end
        clear rn;
        ctx = acorn_training_context(neuron, Cn, []);
    catch err
        fprintf(2, '  FAILED to load %s: %s\n', sd, err.message);
        for i = idx; out{i}.note = err.message; end
        continue;
    end
    for i = idx
        it = items{i};
        col = double(it.review_col);
        try
            if col < 1 || col > ctx.n
                error('review_col %d outside 1..%d', col, ctx.n);
            end
            partner = 0;
            if isfield(it, 'partner_col') && ~isempty(it.partner_col); partner = double(it.partner_col); end
            if partner < 1 || partner > ctx.n; partner = 0; end
            if it.ref_keep == 1; refw = 'reference: keep'; else; refw = 'reference: delete'; end
            if isfield(it, 'ref_motion') && it.ref_motion == 1; refw = [refw, ' (motion)']; end
            label = sprintf('%s -- Neuron %d -- %s', char(it.category), ctx.pos_of_col(col), refw);
            acorn_training_draw_panel(fig, neuron, ctx, col, label, ...
                struct('show_cn', true, 'partner_col', partner));
            png = fullfile(out_dir, char(it.png));
            print(fig, png, '-dpng', '-r80');
            out{i}.shown_pos = ctx.pos_of_col(col);
            out{i}.ok = true;
        catch err
            fprintf(2, '  FAILED item %d (%s col %d): %s\n', i, sd, col, err.message);
            out{i}.note = err.message;
        end
    end
    fprintf('  %s: %d item(s) done\n', sd, numel(idx));
end
close(fig);
R = [out{:}];
fid = fopen(fullfile(out_dir, 'rendered.json'), 'w');
fprintf(fid, '%s', jsonencode(R));
fclose(fid);
fprintf('acorn_training_render_gallery done: %d of %d rendered\n', sum([R.ok]), n);
end
