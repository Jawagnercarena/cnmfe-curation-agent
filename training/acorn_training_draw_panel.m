function acorn_training_draw_panel(fig, neuron, ctx, col, label, opts)
% acorn_training_draw_panel -- draw one candidate the way the review tool does.
%
%   acorn_training_draw_panel(fig, neuron, ctx, col, label, opts)
%
% The single drawing routine shared by the drill, the report and the gallery.
% Panels 1-2 and the trace are the viewNeurons.m layout (full-frame footprint,
% zoom to the centre +/- 2*gSiz, C_raw scaled by max(A) in blue over C in
% red).  With opts.show_cn (default true, needs ctx.Cn_full) a third panel
% shows the correlation image cropped to the same window with this
% candidate's contour in red (and opts.partner_col's in dashed cyan).
% opts.title_color (default 'k'), opts.caption (text strip at the top).
% col is a REVIEW COLUMN (1-based column of neuron.A); label is the title.

if nargin < 6 || isempty(opts); opts = struct(); end
show_cn = getopt(opts, 'show_cn', true) && ~isempty(ctx.Cn_full);
partner = getopt(opts, 'partner_col', 0);
tcolor = getopt(opts, 'title_color', 'k');
caption = getopt(opts, 'caption', '');

set(0, 'CurrentFigure', fig);
clf(fig);
if show_cn; nc = 3; else; nc = 2; end

Acol = neuron.A(:, col) .* ctx.Amask(:, col);
x0 = ctx.ctr(col, 2);
y0 = ctx.ctr(col, 1);
win = [-ctx.gSiz, ctx.gSiz] * 2;

subplot(2, nc, 1, 'Parent', fig);
neuron.image(Acol);
axis equal; axis off;
title(label, 'color', tcolor, 'Interpreter', 'none');

subplot(2, nc, 2, 'Parent', fig);
neuron.image(Acol);
axis equal; axis off;
xlim(x0 + win); ylim(y0 + win);
title('zoom');

if show_cn
    ax = subplot(2, nc, 3, 'Parent', fig);
    imagesc(ctx.Cn_full);
    colormap(ax, 'gray');
    axis equal; axis off; hold(ax, 'on');
    xlim(x0 + win); ylim(y0 + win);
    c1 = contour_of(ctx, col);
    if ~isempty(c1)
        plot(c1(1, :), c1(2, :), 'r', 'LineWidth', 1.5);
    end
    if partner > 0
        c2 = contour_of(ctx, partner);
        if ~isempty(c2)
            plot(c2(1, :), c2(2, :), 'c--', 'LineWidth', 1.2);
        end
    end
    hold(ax, 'off');
    title('Cn (correlation image) + contour');
end

subplot(2, nc, nc + 1:2 * nc, 'Parent', fig);
cla;
plot(ctx.t, full(neuron.C_raw(col, :)) * ctx.Amax(col), 'linewidth', 2);
hold on;
plot(ctx.t, full(neuron.C(col, :)) * ctx.Amax(col), 'r');
hold off;
xlim([ctx.t(1), ctx.t(end)]);
xlabel(ctx.str_xlabel);

if ~isempty(caption)
    annotation(fig, 'textbox', [0.005, 0.935, 0.99, 0.06], 'String', caption, ...
        'Interpreter', 'none', 'FontSize', 9, 'EdgeColor', 'none', ...
        'BackgroundColor', [1, 1, 0.85], 'FitBoxToText', 'off', ...
        'VerticalAlignment', 'middle');
end
end

function v = getopt(o, name, default)
if isfield(o, name) && ~isempty(o.(name))
    v = o.(name);
else
    v = default;
end
end

function c = contour_of(ctx, col)
c = [];
if col >= 1 && col <= numel(ctx.contours) && ~isempty(ctx.contours{col})
    c = ctx.contours{col};
end
end
