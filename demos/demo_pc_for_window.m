%% clear workspace
clear; clc; close all;  
global  d1 d2 numFrame ssub tsub sframe num2read Fs neuron neuron_ds ...
    neuron_full Ybg_weights; %#ok<NUSED> % global variables, don't change them manually

%% select data and map it to the RAM
% nam = '~/Dropbox/github/CNMF_E/demos/data_endoscope.tif';
cnmfe_choose_data;

%% create Source2D class object for storing results and parameters
Fs = 7.5;             % frame rate
ssub = 1;           % spatial downsampling factor
tsub = 2;           % temporal downsampling factor
gSig = 1;           % width of the gaussian kernel, which can approximates the average neuron shape
gSiz = 20;          % maximum diameter of neurons in the image plane. larger values are preferred.
neuron_full = Sources2D('d1',d1,'d2',d2, ... % dimensions of datasets
    'ssub', ssub, 'tsub', tsub, ...  % downsampleing
    'center_psf', 0, ...
    'gSig', gSig,...    % sigma of the 2D gaussian that approximates cell bodies
    'gSiz', gSiz);  % average neuron size (diameter)
neuron_full.Fs = Fs;         % frame rate
dmin = 1; 
merge_thr = [0.3, 0.9, 0]; 
nb = 2; 

% with dendrites or not 
with_dendrites = true;
if with_dendrites
    % determine the search locations by dilating the current neuron shapes
    neuron_full.options.search_method = 'dilate'; 
    neuron_full.options.bSiz = 20;
else
    % determine the search locations by selecting a round area
    neuron_full.options.search_method = 'ellipse';
    neuron_full.options.dist = 5;
end

%% options for running deconvolution 
neuron_full.options.deconv_flag = true;  % set the value as true if you want to do deconvolution 
neuron_full.options.deconv_options = struct('type', 'ar1', ... % model of the calcium traces. {'ar1', 'ar2'}
    'method', 'thresholded', ... % method for running deconvolution {'foopsi', 'constrained', 'thresholded'}
    'optimize_pars', true, ...  % optimize AR coefficients
    'optimize_b', false, ... % optimize the baseline
    'optimize_smin', true);   % optimize the threshold 
%% downsample data for fast and better initialization
sframe=1;						% user input: first frame to read (optional, default:1)
num2read= numFrame;             % user input: how many frames to read   (optional, default: until the end)

tic;
cnmfe_load_data;
fprintf('Time cost in downsapling data:     %.2f seconds\n', toc);

Y = neuron.reshape(Y, 1);       % convert a 3D video into a 2D matrix

%% compute correlation image and peak-to-noise ratio image.
cnmfe_show_corr_pnr;    % this step is not necessary, but it can give you some...
                        % hints on parameter selection, e.g., min_corr & min_pnr

%% initialization of A, C
% parameters
debug_on = false;   % visualize the initialization procedue. 
save_avi = false;   %save the initialization procedure as an avi movie. 
patch_par = [1,1]*1;  % divide the optical field into m X n patches and do initialization patch by patch. It can be used when the data is too large 
K = []; % maximum number of neurons to search within each patch. you can use [] to search the number automatically

min_corr = 0.85;     % minimum local correlation for a seeding pixel
min_pnr = 15;       % minimum peak-to-noise ratio for a seeding pixel
min_pixel = 5^2;      % minimum number of nonzero pixels for each neuron
bd = 1;             % number of rows/columns to be ignored in the boundary (mainly for motion corrected data)
neuron.updateParams('min_corr', min_corr, 'min_pnr', min_pnr, ...
    'min_pixel', min_pixel, 'bd', bd);
neuron.options.nk = 1;  % number of knots for detrending 

% greedy method for initialization
tic;
[center, Cn, pnr] = neuron.initComponents_endoscope(Y, K, patch_par, debug_on, save_avi);
fprintf('Time cost in initializing neurons:     %.2f seconds\n', toc);

% show results
figure;
imagesc(Cn, [0.1, 0.95]);
hold on; plot(center(:, 2), center(:, 1), 'or');
colormap; axis off tight equal;

% sort neurons
[~, srt] = sort(max(neuron.C, [], 2), 'descend');
neuron.orderROIs(srt);
neuron_init = neuron.copy();

%% iteratively update A, C and B
% parameters, merge neurons
display_merge = false;          % visually check the merged neurons
view_neurons = false;           % view all neurons

% parameters, estimate the background
spatial_ds_factor = 2;      % spatial downsampling factor. it's for faster estimation
thresh = 5;     % threshold for detecting frames with large cellular activity. (mean of neighbors' activity  + thresh*sn)

bg_neuron_ratio = 2;  % spatial range / diameter of neurons

% parameters, estimate the spatial components
update_spatial_method = 'hals';  % the method for updating spatial components {'hals', 'hals_thresh', 'nnls', 'lars'}
Nspatial = 5;       % this variable has different meanings: 
                    %1) udpate_spatial_method=='hals' or 'hals_thresh',
                    %then Nspatial is the maximum iteration 
                    %2) update_spatial_method== 'nnls', it is the maximum
                    %number of neurons overlapping at one pixel 
               
% parameters for running iteratiosn 
nC = size(neuron.C, 1);    % number of neurons 

maxIter = 2;        % maximum number of iterations 
miter = 1; 
while miter <= maxIter
    %% merge neurons, order neurons and delete some low quality neurons
    % merge neurons according to temporal correlation 
    cnmfe_quick_merge;              % run neuron merges
    
    % merge neurons based on neuron distances 
    cnmfe_merge_neighbors; 
    
    %% udpate background (cell 1, the following three blocks can be run iteratively)
    % estimate the background
    tic;
    cnmfe_svd_BG; 
    fprintf('Time cost in estimating the background:        %.2f seconds\n', toc);
    % neuron.playMovie(Ysignal); % play the video data after subtracting the background components.
    
    %% update spatial & temporal components
    tic;
    for m=1:2
        %temporal
        neuron.updateTemporal_endoscope(Ysignal);
        cnmfe_quick_merge;              % run neuron merges
        cnmfe_merge_neighbors; 
        %spatial
        neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
        neuron.trimSpatial(0.01, 3); % for each neuron, apply imopen first and then remove pixels that are not connected with the center
%         neuron.compactSpatial(); 
        if isempty(merged_ROI)
            break;
        end
    end
    fprintf('Time cost in updating spatial & temporal components:     %.2f seconds\n', toc);
    
    %% pick neurons from the residual (cell 4).
    if miter==1
        seed_method = 'auto'; % methods for selecting seed pixels {'auto', 'manual'}
        [center_new, Cn_res, pnr_res] = neuron.pickNeurons(Ysignal - neuron.A*neuron.C, patch_par, seed_method, debug_on); % method can be either 'auto' or 'manual'
    end
    
    %% stop the iteration 
    temp = size(neuron.C, 1); 
    if or(nC==temp, miter==maxIter)
        break; 
    else
        miter = miter+1; 
        nC = temp; 
    end
end


%% apply results to the full resolution
if or(ssub>1, tsub>1)
    neuron_ds = neuron.copy();  % save the result
    neuron = neuron_full.copy();
    cnmfe_full;
    neuron_full = neuron.copy();%
end


%% delete some neurons and run CNMF-E iteration
merge_thr = [0.001, 0.75, 0]; 
dmin = 3; 
display_merge = true; 

% sort neurons based on SNR 
snr = var(neuron.C, 0, 2)./var(neuron.C_raw-neuron.C, 0,2); 
[~, ind] = sort(snr, 'descend'); 
neuron.orderROIs(ind); 

neuron.viewNeurons([], neuron.C_raw);
tic;
% cnmfe_update_BG;
cnmfe_svd_BG; 
fprintf('Time cost in estimating the background:        %.2f seconds\n', toc);
%update spatial & temporal components
tic;
for m=1:2
    %temporal
    neuron.updateTemporal_endoscope(Ysignal);
    cnmfe_quick_merge;              % run neuron merges
    cnmfe_merge_neighbors; 
    %spatial
    neuron.updateSpatial_endoscope(Ysignal, Nspatial, update_spatial_method);
    neuron.trimSpatial(0.01, 3); % for each neuron, apply imopen first and then remove pixels that are not connected with the center
%     neuron.compactSpatial(); 
end
fprintf('Time cost in updating spatial & temporal components:     %.2f seconds\n', toc);
cnmfe_svd_BG; 

%% display contours of the neurons
neuron.Coor = neuron.get_contours(0.95); % energy within the contour is 80% of the total 
figure;
Cn = correlation_image(neuron.reshape(Ysignal, 2), 4);
Cn(Cn<0) = 0; 
neuron.Coor = plot_contours(neuron.A, Cn, 0.8, 0, [], neuron.Coor, 2);
axis equal; axis off;
title('contours of estimated neurons');
saveas(gcf, 'contours.pdf'); 
%% %% %% Manually Merge Neurons
cnmfe_manual_merge
%% display neurons
dir_neurons = sprintf('%s%s%s_neurons%s', dir_nm, filesep, file_nm, filesep);
if exist('dir_neurons', 'dir')
    temp = cd();
    cd(dir_neurons);
    delete *;
    cd(temp);
else
    mkdir(dir_neurons);
end
neuron.viewNeurons([], neuron.C_raw, dir_neurons);
close(gcf); 


%% check spatial and temporal components by playing movies
% save_avi = true;
% avi_name = 'play_movie.avi';
% neuron.Cn = Cn;
% center_ac = median(max(neuron.A,[],1)'.*max(neuron.C,[],2)); % the denoised video are mapped to [0, 2*center_ac] of the colormap 
% neuron.runMovie(Ysignal, [0, center_ac], save_avi, avi_name);

%% save video
%kt = 1;     % play one frame in every kt frames
%save_avi = true;
%center_ac = median(max(neuron.A,[],1)'.*max(neuron.C,[],2))/2; % the denoised video are mapped to [0, 2*center_ac] of the colormap 
%cnmfe_save_video;

%% save results
A = neuron.A;
C = neuron.C;
C_raw = neuron.C_raw;
S = neuron.S;
save results A  C C_raw S neuron
% globalVars = who('global');
% eval(sprintf('save %s%s%s_results.mat %s', dir_nm, filesep, file_nm, strjoin(globalVars)));
%% %% %% extract dF/F
%ARE ALL THE BELOW NECESSARY? I SIMPLY ADDED FROM PREVIOUS CNMFE FILE WE WERE
%RUNNING

[neuron.C_df, neuron.Df] = extract_DF_F(Y,neuron.A,neuron.C_raw);

%% write dF/F to file

dlmwrite('C_df.txt',neuron.C_df, 'delimiter',' ')

%%

dlmwrite('C_raw.txt',neuron.C_raw, 'delimiter',' ')
dlmwrite('C.txt',neuron.C, 'delimiter',' ')
dlmwrite('Cnn.txt',Cnn, 'delimiter',' ')
overlapA=neuron.overlapA;
coor=neuron.Coor;

save('Coor.mat','coor')
%% SAVE NEURON AND RESULTS IN WORKSPACE (right click > save as)