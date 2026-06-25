function viewNeurons_dataCollect(obj, ind, C2, folder_nm)
%% view all components and delete components manually. it shows spatial
%   components in the full-frame and zoomed-in view. It also shows temporal
%   components
%% input:
%   ind: vector, indices of components to be displayed, no bigger than the maximum
%       number of neurons
%   C2:  K*T matrix, another temporal component to be displayed together
%       with the esitmated C. usually it is C without deconvolution.
%   folder_nm: string, the folder to output images neuron by neuron.

%% Author: Pengcheng Zhou, Carnegie Mellon University, 2016

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

% obj.delete(sum(obj.A>0, 1)<max(obj.options.min_pixel, 1));

Amask = (obj.A>0); 
ind_trim = false(size(ind));    % indicator of trimming neurons 
ind_del = false(size(ind));     % indicator of deleting neurons
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
figure('position', [100, 100, 1024, 512]);
m=1;
while and(m>=1, m<=length(ind))
    %% full-frame view
    subplot(221);
    obj.image(obj.A(:, ind(m)).*Amask(:, ind(m))); %
    axis equal; axis off;
    if ind_del(m)
        title(sprintf('Neuron %d', ind(m)), 'color', 'r');
    else
        title(sprintf('Neuron %d', ind(m)));
    end
    %% zoomed-in view
    subplot(222);
        obj.image(obj.A(:, ind(m)).*Amask(:, ind(m))); %
%     imagesc(reshape(obj.A(:, ind(m)).*Amask(:,ind(m))), obj.options.d1, obj.options.d2));
    axis equal; axis off;
    x0 = ctr(ind(m), 2);
    y0 = ctr(ind(m), 1);
    xlim(x0+[-gSiz, gSiz]*2);
    ylim(y0+[-gSiz, gSiz]*2);
    
    
    %% temporal components
    subplot(2,2,3:4);cla;
    if ~isempty(C2)
        plot(t, C2(ind(m), :)*max(obj.A(:, ind(m))), 'linewidth', 2); hold on;
        plot(t, obj.C(ind(m), :)*max(obj.A(:, ind(m))), 'r');
    else
        
        plot(t, obj.C(ind(m), :)*max(obj.A(:, ind(m))));
    end
    xlim([t(1), t(end)]); 
    xlabel(str_xlabel);
    
    %% save images
    if save_img
        saveas(gcf, sprintf('neuron_%d.png', ind(m)));
        m = m+1;
    else
        fprintf('Neuron %d, keep(k, default)/delete(d)/split(s)/trim(t)/trim cancel(tc)/delete all(da)/backward(b)/end(e):    ', ind(m));

        temp = input('', 's');
        if temp=='d'
            % collect data on neuron (why are we deleting?)
            fileID = fopen("data.txt","a"); % open file to add data
            % find info about raw trace
            rawTrace = full(C2(ind(m), :)*max(obj.A(:, ind(m)))); 
            mean_raw = mean(rawTrace);
            median_raw = median(rawTrace);
            max_raw = max(rawTrace);
            min_raw = min(rawTrace);
            std_raw = std(rawTrace);
            iqr_raw = iqr(rawTrace);
            positive_raw = sum(rawTrace > 0)/length(rawTrace);
            skew_raw = sum(rawTrace > mean_raw)/length(rawTrace);
            % find info about transients
            transientTrace = full(obj.C(ind(m), :)*max(obj.A(:, ind(m))));
            mean_trans = mean(transientTrace);
            median_trans = median(transientTrace);
            max_trans = max(transientTrace);
            min_trans = min(transientTrace);
            std_trans = std(transientTrace);
            iqr_trans = iqr(transientTrace);
            skew_trans = sum(transientTrace > mean_raw)/length(transientTrace);
            % find info about spike timing
            spike_time = sum(obj.S(ind(m),:) > 0)/length(obj.S(ind(m),:));
            % find info about neuron ROI
            Coor = obj.get_contours(0.8, ind(m));
            contour = Coor{1,1};
            warning('off','all'); % surpress warning from this next command
            poly = polyshape(contour(1,:), contour(2,:));
            warning('on','all');
            area_ROI = area(poly);
            perim_ROI = perimeter(poly);
            ratio_ROI = area_ROI/perim_ROI/perim_ROI; % dimensionless measure of circularity, should be less than 1/4pi = 0.079
            % print this info out
            fprintf("0 %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", mean_raw, median_raw, max_raw, min_raw, std_raw, iqr_raw, positive_raw, skew_raw, mean_trans, median_trans, max_trans, min_trans, std_trans, iqr_trans, skew_trans, spike_time, area_ROI, perim_ROI, ratio_ROI);
            fprintf(fileID, "0 %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", mean_raw, median_raw, max_raw, min_raw, std_raw, iqr_raw, positive_raw, skew_raw, mean_trans, median_trans, max_trans, min_trans, std_trans, iqr_trans, skew_trans, spike_time, area_ROI, perim_ROI, ratio_ROI);
            fclose(fileID); % close file
            ind_del(m) = true;
            m = m+1;
        elseif strcmpi(temp, 'b')
            m = m-1;
        elseif strcmpi(temp, 'da')
            ind_del(m:end) = true;
            break;
        elseif strcmpi(temp, 'k')
            ind_del(m) = false;
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
            m = m + floor(str2double(temp)); 
            fprintf('jump to neuron %d / %d', m, length(ind)); 
        else
           % collect data on neuron (why are we keeping?)
            fileID = fopen("data.txt","a"); % open file to add data

            % find info about raw trace
            rawTrace = full(C2(ind(m), :)*max(obj.A(:, ind(m)))); 
            mean_raw = mean(rawTrace);
            median_raw = median(rawTrace);
            max_raw = max(rawTrace);
            min_raw = min(rawTrace);
            std_raw = std(rawTrace);
            iqr_raw = iqr(rawTrace);
            positive_raw = sum(rawTrace > 0)/length(rawTrace);
            skew_raw = sum(rawTrace > mean_raw)/length(rawTrace);

            % find info about transients
            transientTrace = full(obj.C(ind(m), :)*max(obj.A(:, ind(m))));
            mean_trans = mean(transientTrace);
            median_trans = median(transientTrace);
            max_trans = max(transientTrace);
            min_trans = min(transientTrace);
            std_trans = std(transientTrace);
            iqr_trans = iqr(transientTrace);
            skew_trans = sum(transientTrace > mean_raw)/length(transientTrace);

            % find info about spike timing
            spike_time = sum(obj.S(ind(m),:) > 0)/length(obj.S(ind(m),:));

            % find info about neuron ROI
            Coor = obj.get_contours(0.8, ind(m));
            contour = Coor{1,1};
            warning('off','all'); % surpress warning from this next command
            poly = polyshape(contour(1,:), contour(2,:));
            warning('on','all');
            area_ROI = area(poly);
            perim_ROI = perimeter(poly);
            ratio_ROI = area_ROI/perim_ROI/perim_ROI; % dimensionless measure of circularity, should be less than 1/4pi = 0.079

            % print this info out
            fprintf("1 %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", mean_raw, median_raw, max_raw, min_raw, std_raw, iqr_raw, positive_raw, skew_raw, mean_trans, median_trans, max_trans, min_trans, std_trans, iqr_trans, skew_trans, spike_time, area_ROI, perim_ROI, ratio_ROI);
            fprintf(fileID, "1 %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", mean_raw, median_raw, max_raw, min_raw, std_raw, iqr_raw, positive_raw, skew_raw, mean_trans, median_trans, max_trans, min_trans, std_trans, iqr_trans, skew_trans, spike_time, area_ROI, perim_ROI, ratio_ROI);
            fclose(fileID); % close file

            m = m+1;
        end
    end
end
if save_img
    cd(cur_cd);
else
    obj.A(:, ind(ind_trim)) = obj.A(:,ind(ind_trim)).*Amask(:, ind(ind_trim)); 
    obj.delete(ind(ind_del));
%     obj.Coor = obj.get_contours(0.9);
end

