 function viewNeuronsVideo(obj, ind, C2, folder_nm)
%% view all components and delete components manually. 
% It shows the video in the full-frame and the spatial components in a zoomed-in view. 
% It also shows temporal components.
%% input:
%   ind: vector, indices of components to be displayed, no bigger than the maximum
%       number of neurons
%   C2:  K*T matrix, another temporal component to be displayed together
%       with the esitmated C. usually it is C without deconvolution.
%   folder_nm: string, the folder to output images neuron by neuron.
%% Author: Pengcheng Zhou, Carnegie Mellon University, 2016
%% Modified by: Aadith Vittala, Rice University, 2019
    global ssub % use the global ssub variable for scaling of the contours
    global MOTION_DELETE_FP; % motion-delete footprint accumulator (see viewNeurons)
    if ~exist('ind', 'var') || isempty(ind)
        % display all neurons if ind is not specified.
        ind = 1:size(obj.A, 2);
    elseif ind==-1 
        ind = size(obj.A,2):-1:1; 
    end
    if ~exist('C2', 'var'); C2=[]; end

    if exist('folder_nm', 'var')&&(~isempty(folder_nm))
        % create a folder to save images
        save_img = true;
        cur_cd = cd();
        if ~exist(folder_nm, 'dir'); mkdir(folder_nm);
        else
            fprintf('The folder has been created and old results will be overwritten. \n');
        end
        cd(folder_nm);
    else
        save_img = false;
    end

    Amask = (obj.A>0); 
    ind_trim = false(size(ind));    % indicator of trimming neurons
    ind_del = false(size(ind));     % indicator of deleting neurons
    ind_motion = false(size(ind));  % indicator of motion-delete neurons (subset of ind_del)
    ctr = obj.estCenter();      %neuron's center
    gSiz = obj.options.gSiz;        % maximum size of a neuron

    % time
    T = size(obj.C, 2);
    t = 1:T;
    if ~isnan(obj.Fs)
        t = t/obj.Fs;
        str_xlabel = 'Time (Sec.)';
    else
        str_xlabel = 'Frame';
    end

    %% start viewing neurons
    fig_video = figure('position', [100, 100, 1024, 512]);
    m=1;
    fprintf("Showing neurons..\n");

    function scrollCallback(src,evt,im,video,num_frames)
        index = round(src.Value*(num_frames-1))+1; % find frame in the video
        im.CData = video(:,:,index); % replace image with correct timeframe in video
        t_index = scrollbar.Value*t(end);
        axes = subplot(2,2,3:4);
        delete(l);
        l = plot([t_index, t_index], get(gca, 'ylim'), 'y');
    end
    function clickCallback(src,evt,allCoor,zoom)
        pos = evt.IntersectionPoint(1:2); % position of click
        neuronClicked = false;
        for j = 1:size(allCoor) % check if click in contour and print if so
            con = allCoor{j,1};
            if  inpolygon(pos(1),pos(2),con(1,:)*1,im.YData(2)-con(2,:)*1)
           % if inpolygon(pos(1),pos(2),con(1,:)*zoom,im.YData(2)-con(2,:)*zoom) 
                neuronClicked = true;
                fprintf("\nClicked on neuron %i", j); % print which neuron
            end
        end
        if neuronClicked
            fprintf('\nNeuron %d, keep(k, default)/delete(d)/MOTION delete(m)/split(s)/trim(t)/trim cancel(tc)/delete all(da)/backward(b)/end(e)/jump to(#):    ', ind(m));
        end
    end
    video_struct = load(obj.options.name); % load video from the .mat file
    video = flip(video_struct.Y); % extract video and flip over y-axis
    yflip = size(video, 1); % image height (== im.YData(2)); used to flip contour
                            % y-coords WITHOUT touching the im handle, which can go
                            % invalid while jumping between neurons and abort the review

    subplot(221); 
    hold on;
    %colormap(gray); % plot in grayscale
    im = imagesc(video(:,:,1)); % plot first image in video
    allCoor = obj.get_contours(0.8,ind); % find all contours
    im.ButtonDownFcn = {@clickCallback,allCoor,ssub/obj.options.ssub};
    num_frames = video_struct.Ysiz(3); % find total number of frames in video
    axis equal; axis off;
    axes = subplot(2,2,3:4);cla;
    scroll_pos = [axes.Position(1),axes.Position(2)+0.37,axes.Position(3),0.04]; % position of scroll bar
    scrollbar = uicontrol('Style','slider','Units','normalized','Callback',{@scrollCallback,im,video,num_frames},'Position',scroll_pos,'SliderStep',[1/num_frames,0.01]);
    scrollbar.Value = 0; % start scrollbar at zero
    
    % A whole pass has been lost here before (see the contour note below): every
    % decision the reviewer makes lives in ind_del/ind_motion/ind_trim and is only
    % applied AFTER this loop, so any error escaping the loop discards the lot and
    % takes CNMFe_final_save down with it, back to the last checkpoint. Catch here
    % and fall through to the apply block instead: a partial pass is recoverable
    % (the caller offers another pass), a discarded one is not.
    try
        while and(m>=1, m<=length(ind))
            % Reviewers are told 'Close the window when done', and doing that mid-pass
            % used to throw on the cached im/scrollbar handles below. Treat a missing
            % window as 'e' (end). This has to run BEFORE any drawing: subplot() on a
            % closed figure silently opens a fresh empty one and the pass limps on in it.
            if ~isvalid(fig_video) || ~isvalid(im) || ~isvalid(scrollbar)
                fprintf('\nVideo window closed -- ending this pass. %d delete(s) and %d motion tag(s) so far are kept.\n', sum(ind_del), sum(ind_motion));
                break;
            end

            %% full-frame view
            subplot(221); % make title for this view
            if ind_del(m)
                title(sprintf('Neuron %d', ind(m)), 'color', 'r');
            else
                title(sprintf('Neuron %d', ind(m)));
            end
            %% zoomed-in view of spatial component
            subplot(222);
            obj.image(obj.A(:, ind(m)).*Amask(:, ind(m)));
            axis equal; axis off;
            x0 = ctr(ind(m), 2);
            y0 = ctr(ind(m), 1);
            xlim(x0+[-gSiz, gSiz]*2);
            ylim(y0+[-gSiz, gSiz]*2);

            %% temporal components
            axes = subplot(2,2,3:4);cla;
            if ~isempty(C2)
                plot(t, C2(ind(m), :)*max(obj.A(:, ind(m))), 'linewidth', 2); hold on;
                plot(t, obj.C(ind(m), :)*max(obj.A(:, ind(m))), 'r'); hold on;
            else

                plot(t, obj.C(ind(m), :)*max(obj.A(:, ind(m))));
            end
            xlim([t(1), t(end)]); 
            xlabel(str_xlabel);    

            %% video with contours
            subplot(221);
            % Draw contours defensively. Two things here have aborted whole reviews
            % (losing the reviewer's labels): a degenerate all-zero footprint yields an
            % EMPTY contour (indexing contour(1,:) then errors), and the cached image
            % handle im can go INVALID while jumping between neurons (im.YData(2) then
            % errors). So: skip empty contours, use the numeric image height yflip
            % instead of im.YData(2), and never let a draw glitch propagate out.
            try
                allCoor = obj.get_contours(0.8,ind); % find all contours
                for i = 1:numel(allCoor) % extract and plot all contours in blue
                    contour = allCoor{i,1};
                    if isempty(contour); continue; end % degenerate footprint -> no contour
                    plot(contour(1,:)*2/obj.options.ssub,yflip-contour(2,:)*2/obj.options.ssub,"-k","PickableParts","none");
                end
                Coor = obj.get_contours(0.8,ind(m)); % find the contours for this neuron
                if ~isempty(Coor) && ~isempty(Coor{1,1})
                    contour = Coor{1,1}; % extract coordinates
                    plot(contour(1,:)*2/obj.options.ssub,yflip-contour(2,:)*2/obj.options.ssub,"-r","PickableParts","none");
                end
            catch ME
                fprintf(2, 'Contour draw skipped for neuron %d (%s); review continues.\n', ind(m), ME.message);
            end

            %% move scrollbar to time of max value
            [maxVal, maxInd] = max(obj.C(ind(m),:));
            scrollbar.Value = (maxInd-1)/(num_frames-1);
            im.CData = video(:,:,maxInd);
            t_index = scrollbar.Value*t(end);
            axes = subplot(2,2,3:4);
            l = plot([t_index, t_index], get(gca, 'ylim'), 'y'); hold on;
        
            %% save images
            if save_img
                saveas(gcf, sprintf('neuron_%d.png', ind(m)));
                m = m+1;
            else
                fprintf('Neuron %d, keep(k, default)/delete(d)/MOTION delete(m)/split(s)/trim(t)/trim cancel(tc)/delete all(da)/backward(b)/end(e)/jump to(#):    ', ind(m));
                temp = input('', 's');
                if temp=='d'
                    ind_del(m) = true;
                    ind_motion(m) = false;
                    m = m+1;
                elseif strcmpi(temp, 'm')
                    ind_del(m) = true;
                    ind_motion(m) = true;
                    m = m+1;
                elseif strcmpi(temp, 'b')
                    m = m-1;
                elseif strcmpi(temp, 'da')
                    ind_del(m:end) = true;
                    break;
                elseif strcmpi(temp, 'k')
                    ind_del(m) = false;
                    ind_motion(m) = false;
                    m= m+1;
                elseif strcmpi(temp, 's')
                    try
                        subplot(222);
                        temp = imfreehand();
                        tmp_ind = temp.createMask();
                        tmpA = obj.A(:, ind(m));
                        obj.A(:, end+1) = tmpA.*tmp_ind(:);
                        obj.C(end+1, :) = obj.C(ind(m), :);
                        obj.A(:, ind(m)) = tmpA.*(1-tmp_ind(:));
                        obj.S(end+1, :) = obj.S(ind(m), :);
                        obj.C_raw(end+1, :) = obj.C_raw(ind(m), :);
                        obj.P.kernel_pars(end+1, :) = obj.P.kernel_pars(ind(m), :);
                    catch
                        fprintf('the neuron was not split\n');
                    end
                elseif strcmpi(temp, 't')
                    try
                        subplot(222);
                        temp = imfreehand();
                        tmp_ind = temp.createMask();
                        Amask(:, ind(m)) = tmp_ind(:);
                        ind_trim(m) = true; 
                    catch
                        fprintf('the neuron was not trimmed\n');
                    end
                elseif strcmpi(temp, 'tc')
                        Amask(:, ind(m)) = (obj.A(:, ind(m)) > 0);
                        ind_trim(m) = false; 
                elseif strcmpi(temp, 'e')
                    break;
                elseif ~isnan(str2double(temp))
                    % "jump to(#)" is the neuron number shown in the prompt, so look it up in ind
                    % rather than assigning it to m -- the two coincide only because every caller
                    % passes ind = [] (-> 1:N). Numbers that are not neurons in this pass are now
                    % refused: assigning them let "0", "-3", "201" and "Inf" push m past the loop
                    % bounds, silently ENDING the pass with every unreviewed neuron left at its
                    % default (keep), and "3+4i" threw. The two viewers disagreed here as well --
                    % one added the number to m, the other replaced it -- so both now do the same.
                    jump_to = find(ind == str2double(temp), 1);
                    if isempty(jump_to)
                        fprintf('There is no neuron %s in this pass (neurons %d-%d) -- staying on neuron %d.\n', ...
                            strtrim(temp), min(ind), max(ind), ind(m));
                    else
                        m = jump_to;
                        fprintf('jump to neuron %d (%d / %d)\n', ind(m), m, length(ind));
                    end
                else
                    m = m+1;
                end
            end
        end
    catch ME
        fprintf(2, ['\nVideo review interrupted: %s\n' ...
                    'Ending this pass; %d delete(s) and %d motion tag(s) made so far are kept.\n'], ...
            ME.message, sum(ind_del), sum(ind_motion));
    end
    % Review over: drop both nested-function callbacks. Each is a handle into THIS
    % workspace, so while either one survives the figure pins video_struct.Y and
    % video -- two full copies of the session video -- for as long as the window is
    % left open, and one figure is created per pass. A stray click on a finished
    % figure also fprintf's the review prompt onto the >> prompt, where the next
    % Enter feeds it to the parser as a command ("Unexpected MATLAB expression").
    % The window is left OPEN on purpose so the reviewer can still see which cell
    % they were on; the scrollbar is greyed out rather than left as a live-looking
    % control that silently does nothing.
    if isvalid(im); im.ButtonDownFcn = []; end
    if isvalid(scrollbar); scrollbar.Callback = []; scrollbar.Enable = 'off'; end

    if save_img
        cd(cur_cd);
    else
        if any(ind_motion)
            % Snapshot footprints of motion-tagged neurons BEFORE deletion so their
            % identity can be matched back to the review candidates in CNMFe_final_save.
            MOTION_DELETE_FP = [MOTION_DELETE_FP, full(obj.A(:, ind(ind_motion)))];
        end
        obj.A(:, ind(ind_trim)) = obj.A(:,ind(ind_trim)).*Amask(:, ind(ind_trim));
        obj.delete(ind(ind_del));
    end
end
