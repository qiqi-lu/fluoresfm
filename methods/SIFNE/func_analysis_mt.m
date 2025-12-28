% ANALYSIS WITHOUT GUI.
% The codes are extracted from other GUI files.
% Can only analysis single image.
% @qiqilu
function func_analysis_mt(imgpath)
%% PARAMETER SETTINGS
% clc
% clear
disp('### Parameter setting ...')

% imgpath = "E:\qiqilu\datasets\SIFNE\unzip\SIFNE\MT.tiff";           % path of image
% maskpath = "E:\qiqilu\datasets\SIFNE\unzip\SIFNE\MT_cell_mask.tiff"; % path of hand-drawn mask using ImageJ

% imgpath = "E:\qiqilu\datasets\RCAN3D\transformed\C2S_MT\test\channel_0\STED_2d\29_0.tif";
% imgpath = "E:\qiqilu\datasets\RCAN3D\transformed\C2S_MT\test\channel_0\confocal_2d\29_0.tif";
% imgpath = "E:\qiqilu\Project\2024 Foundation model\code\results\predictions\rcan3d-c2s-mt-sr\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16\29_0.tif";

% id_sample = 46;
% imgpath = "E:\qiqilu\datasets\BioSR\transformed\MTs\test\channel_0\SIM\"+num2str(id_sample)+".tif";
% imgpath = "E:\qiqilu\datasets\BioSR\transformed\MTs\test\channel_0\WF_noise_level_3_up2\"+num2str(id_sample)+".tif";
% imgpath = "E:\qiqilu\Project\2024 Foundation model\code\results\predictions\biosr-mt-sr-3\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16\"+num2str(id_sample)+".tif";
maskpath = "";

disp('Image Path: '+imgpath)
disp('### Done.')

%% Path setting
disp('### Set path ...')

R = 10; % radius of filter, default 10.
NofOrientations_FT = 20; % number of filter orientations
enable_preview     = 0;  % whether to preview results (show figures)
enable_normalize = 0;

[path_root, filename, ~] = fileparts(imgpath);
path_save_root = fullfile(path_root,filename+'_analysis');

path_data = fullfile(path_save_root,'data');
path_result = fullfile(path_save_root,'result');
path_settings = fullfile(path_save_root, 'UserSettings');

dirs = {path_data, path_result, path_settings};
for i = 1: numel(dirs)
    if ~exist(dirs{i}, 'dir')
        mkdir(dirs{i});
    end
end

disp('Save analysis output to: ' + path_save_root)
disp('### Done.')

%% LoadImg
disp('### Load image ...')

% save data\imgpath.mat imgpath
save(fullfile(path_data,'imgpath.mat'), 'imgpath'); % path of input image
% read image
OriginImg = imread(imgpath); 
OriginImg = squeeze(OriginImg);
% convert to gray image
if length(size(OriginImg))==3
    OriginImg = rgb2gray(OriginImg);
end
% rescale to 0-1
if isa(OriginImg,'single') || isa(OriginImg,'double')
    OriginImg = rescale(OriginImg);
end

if enable_normalize == 1
    OriginImg = normalize_clip(OriginImg);
end

OriginImg = imadjust(im2uint8(OriginImg));

% save data\OriginImg.mat OriginImg
save(fullfile(path_data,'OriginImg.mat'), 'OriginImg');

% show loaded image
if enable_preview == 1
    figure('name', 'Image loaded');
    imshow(OriginImg);
    axis off;
end

disp('Shape of image:')
disp(size(OriginImg))
disp('### Done.')

%% Preview FT
% The image enhancement methd used in this algorithm is line and
% orientation filter transform (LFT and OFT). The user should define the
% radius and number of rotations of the scanning line segmenta.
% Preview of the filter dimension of LFT adn OFT

disp('### Preview filter ...')

warning off;
close(figure(1));

% load data\OriginImg.mat;
load(fullfile(path_data,'OriginImg.mat'))

I = OriginImg;
[H, W] = size(I);
mask = uint8(zeros(H+R+R,W+R+R));
mask(R+1:H+R,R+1:W+R) = I;
ROI_Mask = ones(size(mask));

% save data\ROI_Mask.mat ROI_Mask;
% save data\R.mat R;
% save data\NofOrientations_FT.mat NofOrientations_FT;

save(fullfile(path_data,'ROI_Mask.mat'),'ROI_Mask')
save(fullfile(path_data,'R.mat'),'R')
save(fullfile(path_data,'NofOrientations_FT.mat'),'NofOrientations_FT')

[H, W] = size(mask);
AngleList = 0:pi/NofOrientations_FT:pi-pi/NofOrientations_FT;

PtsSide1 = [(R*cos(AngleList)+W/2-6)'        (R*sin(AngleList)+H/2-6)'];
PtsSide2 = [(R*cos(AngleList+pi)+W/2-6)'   (R*sin(AngleList+pi)+H/2-6)'];

PtsAll = [PtsSide1;PtsSide2;PtsSide1(1,1)  PtsSide1(1,2)];

% show filter size
if enable_preview == 1
    figure('name','Check Parameters before Filter Transform');
    imshow(mask);hold on;axis off;
    plot(PtsAll(:,2),PtsAll(:,1),'color','r');
    for i = 1:length(AngleList)
        plot([PtsSide1(i,2)  PtsSide2(i,2)],[PtsSide1(i,1)  PtsSide2(i,1)],'color','r');
    end
end

disp('### Done.')

%% GET CELL MASK
% choose region of interest. This region will be used to dfien the cell
% boundary and calculate the distance map in the analysis section. The user
% should choose the ROI carefully.
% Here, a mask drawn using ImageJ can be used, which should be specified at
% the beginning parameter setting part.

disp('### Load cell mask ...')

% save data\R.mat R;
% save data\NofOrientations_FT.mat NofOrientations_FT;
% load data\OriginImg;

% get ROI_mask use mouse click [YOU CAN USE THIS]
% load(fullfile(path_data,"OriginImg.mat"))
% close(figure(1));
% I = zeros(size(OriginImg,1)+2*R,  size(OriginImg,2)+2*R);
% I(R+1:size(I,1)-R,  R+1:size(I,2)-R) = OriginImg;
% 
% figure('name','Please Select the Region of Interest');
% imshow(mat2gray(I));
% 
% ROI_Mask = roipoly;
% save data\ROI_Mask.mat ROI_Mask;
% save(fullfile(path_data,"ROI_Mask.mat"),'ROI_Mask')
% msgbox('ROI Selected !');
% 
% close(figure(1));

% get ROI_mask from file
if exist(maskpath,'dir')
    mask_ori = imread(maskpath);
    mask_ori = logical(mask_ori);
    ROI_Mask = zeros(size(OriginImg,1)+2*R,  size(OriginImg,2)+2*R);
    ROI_Mask(R+1:size(ROI_Mask,1)-R,  R+1:size(ROI_Mask,2)-R)= mask_ori;
else
    if maskpath == ""
        ROI_Mask = ones(size(OriginImg,1)+2*R,  size(OriginImg,2)+2*R);
    else
        disp('Need a mask or let maskpath to be "".')
    end
end

% save data\ROI_Mask.mat ROI_Mask;
save(fullfile(path_data,'ROI_Mask.mat'),'ROI_Mask')

disp('### Done.')

%% LFT_OFT
% Line and orientation filter transform
disp('### Run line and orientation filter trasnforms - use mex function ...')

close(figure(1));

% load data\R.mat R;
% load data\NofOrientations_FT.mat NofOrientations_FT;
% load data\OriginImg.mat;
% load data\ROI_Mask;

[H, W] = size(OriginImg);
OriginImg_Margin = uint8(zeros(H+R+R,W+R+R));
OriginImg_Margin(R+1:H+R,R+1:W+R) = OriginImg;

[OFT_Img, LFT_Img, LFT_Orientations] = LFT_OFT_mex(double(OriginImg_Margin),double(R),double(NofOrientations_FT),double(ROI_Mask));
disp('Transformation Done !');

% save data\OFT_Img.mat OFT_Img;
% save data\LFT_Img.mat LFT_Img;
% save data\LFT_Orientations.mat LFT_Orientations;

save(fullfile(path_data,"OFT_Img.mat"),'OFT_Img')
save(fullfile(path_data,"LFT_Img.mat"),'LFT_Img')
save(fullfile(path_data,"LFT_Orientations.mat"),'LFT_Orientations')

if enable_preview == 1
    figure('name','Check the Enhanced Image');
    imshow(mat2gray(OFT_Img));
    axis off;
end

disp('Done.')

%% Segmentation
% automatically calculate the threshold for binarizing the enhanced image
% whose intesities has been normalized to 1.
% the default threshold is 1.42 times the value calculated using Otsu's
% method.

disp('### Segentation ...')

warning off;

% load data\OFT_Img;
% load data\OriginImg;
% load data\R;

DefaultFactor = 1.42;

I = mat2gray(OFT_Img);
t = DefaultFactor*graythresh(I); % Otsu's threshold

if t>=1
    t = graythresh(I);
end

% BW = im2bw(I,t);
BW = imbinarize(I);
RawSke = bwmorph(BW,'thin',Inf); % get the skeleton of the segmentation
[x, y] = find(RawSke==1); % get the coordinates of all pixels in the skeleton

ThreshOpt1Edit=t;

disp(['The Otsus Threshold Is ',num2str(graythresh(I))]);

Allpts = [x y];
AllFragments = RawSke;

% save data\Allpts.mat Allpts;
% save data\RawSke.mat RawSke;
% save data\AllFragments.mat AllFragments;

save(fullfile(path_data,"Allpts.mat"),"Allpts")
save(fullfile(path_data,"RawSke.mat"),"RawSke")
save(fullfile(path_data,"AllFragments.mat"),"AllFragments")

% show skeleton of binarized image
if enable_preview == 1
    close(figure(1));
    figure('name','Check Segmented Image (Left) and Extracted Skeleton (Right)');
    subplot(1,2,1);
    imshow(BW);axis off; title('Segmented Image');
    subplot(1,2,2);
    imshow(mat2gray(OriginImg));hold on;
    plot(y-R,x-R,'r.');axis off; 
    title('Original Image with Extracted Skeleton');
    % scrsz = get(0,'ScreenSize');
    % set(figure(1),'Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)])
end

disp('### Done.')

%% Junction removal
% To create the pool of minimal linear filament fragments, regions of
% junctions should be removed.

% remove margin information below

disp('### Remove junction ...')

% load data\RawSke.mat;
% load data\OriginImg.mat;
% load data\R.mat;

AllFragments = RawSke;
NOptsMargin = R;
DeleteList = [];

% remove the boader margins
AllFragments(1:NOptsMargin,:) = 0;
AllFragments((end-NOptsMargin):end,:) = 0;
AllFragments(:,1:NOptsMargin) = 0;
AllFragments(:,(end-NOptsMargin):end) = 0;

[x, y] = find(AllFragments==1);
Allpts = [x y];
% remove margin information above

% remove crossing points, a N-by-N region located at the base point will be removed
SizofJuncEdit = 7;
R_Junc = floor((SizofJuncEdit-1)/2);
Size_Junc = SizofJuncEdit;


% compute neighborhood connections for every skeleton pixel
% ----------
% N1 N4 N6
% N2    N7
% N3 N5 N8
% ----------
N1 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)-1,Allpts(:,2)-1));
N2 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)-1,Allpts(:,2)));
N3 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)-1,Allpts(:,2)+1));
N4 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1),Allpts(:,2)-1));
N5 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1),Allpts(:,2)+1));
N6 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)+1,Allpts(:,2)-1));
N7 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)+1,Allpts(:,2)));
N8 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)+1,Allpts(:,2)+1));

% count neighbors to find crossing points
N = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8;

CrPts = Allpts(find(N>=3),1:2); % coordinates of the crossing points

% remove crossing points (junctions)
h = waitbar(0,'Removing crossing points');
for i = 1:size(CrPts,1)
    waitbar(i/size(CrPts,1),h);
    % set the block centered at the crossing point to 0
    AllFragments(CrPts(i,1)-R_Junc:CrPts(i,1)+R_Junc,CrPts(i,2)-R_Junc:CrPts(i,2)+R_Junc) = 0;
end

close(h);
[x, y] = find(AllFragments==1); % get the remianing skeleton pixels
Allpts = [x y];

% recompute neighborhood counts to find isolated/single points
N1 = AllFragments(sub2ind(size(AllFragments),x-1,y-1));
N2 = AllFragments(sub2ind(size(AllFragments),x-1,y));
N3 = AllFragments(sub2ind(size(AllFragments),x-1,y+1));
N4 = AllFragments(sub2ind(size(AllFragments),x,y-1));
N5 = AllFragments(sub2ind(size(AllFragments),x,y+1));
N6 = AllFragments(sub2ind(size(AllFragments),x+1,y-1));
N7 = AllFragments(sub2ind(size(AllFragments),x+1,y));
N8 = AllFragments(sub2ind(size(AllFragments),x+1,y+1));

N = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8;

SinglePts = Allpts(find(N==0),1:2); % wihite neighbors as 

% remove single points
h = waitbar(0,'Removing single points');
for i = 1:size(SinglePts,1)
    waitbar(i/size(SinglePts,1),h);
    AllFragments(SinglePts(i,1),SinglePts(i,2)) = 0;
end

close(h);
[x, y] = find(AllFragments==1); % get remaining points
Allpts = [x y];
[L, num] = bwlabel(AllFragments,8); % label connested components using 8-connectivity. L is labeld matrix, num is the detected fagments.

RawCrPts = CrPts;

% save data\L.mat L num;
% save data\AllFragments.mat AllFragments;
% save data\Allpts.mat Allpts;
% save data\Size_Junc.mat Size_Junc;
% save data\RawCrPts.mat RawCrPts;

save(fullfile(path_data,"L.mat"),"L","num")
save(fullfile(path_data,"AllFragments.mat"),"AllFragments")
save(fullfile(path_data,"Allpts.mat"),"Allpts")
save(fullfile(path_data,"Size_Junc.mat"),"Size_Junc")
save(fullfile(path_data,"RawCrPts.mat"),"RawCrPts")

% show
if enable_preview == 1
    close(figure(1));
    figure('name','Individual Filamentous Fragments');
    imshow(mat2gray(OriginImg));hold on;plot(y-R,x-R,'r.');axis off;
end
% remove clusters of single points

disp('### Done.')

%% Remove short line segments
% It is suggested to remove some short filament fragments primarily
% generated from noise.

disp('### Remove short line segments ...')

% remove very short fragment

% load data\AllFragments.mat;
% load data\R.mat
% load data\OriginImg.mat
% load data\Allpts.mat;
% load data\L.mat;

MinNofPixels = 6;

MIN_FragmentLength = MinNofPixels;
AllFragments = bwareaopen(AllFragments, MIN_FragmentLength); % delete small segments
[L, num] = bwlabel(AllFragments,8); % relabel fragments
[x, y] = find(AllFragments==1);
Allpts = [x y];

% save data\AllFragments.mat AllFragments;
% save data\Allpts.mat Allpts;
% save data\L.mat L num;

save(fullfile(path_data,"AllFragments.mat"),"AllFragments")
save(fullfile(path_data,"Allpts.mat"),"Allpts")
save(fullfile(path_data,"L.mat"),"L","num")

if enable_preview == 1
    close(figure(1));
    figure('name','Filtered Skeleton (Short Filaments Removed)');
    imshow(mat2gray(OriginImg));hold on;
    plot(y-R,x-R,'r.');axis off;
end
% remove very short fragment

disp('### Done.')

%% (Optional) Iterative Extraction of Linear Fragments
% An iterative extraction of filament fragments will significantly recover
% undetected linear structures, especially in highly complex filament
% networks.

disp('### Iteratively extract linear fragments ...')

% load data\OriginImg.mat;
% load data\AllFragments.mat;
% load data\OFT_Img.mat;
% load data\R.mat;
% load data\ROI_Mask.mat;
% load data\NofOrientations_FT.mat;

Thresh = ThreshOpt1Edit;
JuncSize = SizofJuncEdit;
MIN_FragmentLength = MinNofPixels;

IterList = 1;       % choose the number of additional iterations from 1 to 5.

iteration = IterList;
Iter_RemoveR = 3;   % unit: pixels
R_Junc = (JuncSize-1)/2;

AllFragments = IterGenFragment(OriginImg, ...
    AllFragments, ...
    iteration, ...
    R, ...
    ROI_Mask, ...
    NofOrientations_FT, ...
    Iter_RemoveR, ...
    Thresh, ...
    R_Junc, ...
    MIN_FragmentLength);

% -------------------------------------------------------------------------
[L, num] = bwlabel(AllFragments); % relabel fragments
% -------------------------------------------------------------------------
[x, y] = find(AllFragments==1);
Allpts = [x y];

% msgbox('Iterative Extraction of Fragments Done !');
disp('Iterative Extraction of Fragments Done !')

if enable_preview == 1
    close(figure(1));
    figure('name','Ultimate Fragments After Iterative Processing');
    imshow(mat2gray(OriginImg));hold on;
    plot(y-R,x-R,'r.');axis off;
end

% save data\AllFragments.mat AllFragments;
% save data\Allpts.mat Allpts;
% save data\L.mat L num;

save(fullfile(path_data,"AllFragments.mat"),"AllFragments")
save(fullfile(path_data,"Allpts.mat"),"Allpts")
save(fullfile(path_data,"L.mat"),"L","num")

disp('### Done.')

%% Tip Registration
% to register the propagation direciton of each tip.
% To increase the computational speed, the program is configured for
% parallel computing.
% - None: No parallel computing is needed.
% - Half: Use half of the cores.
% - Max: Use all cores.

disp('### Register tips ...')

% load data\AllFragments.mat;
% load data\Allpts.mat;
% load data\L.mat;
% load data\R.mat;

LL = L;             % labeled image of fragments (each has a numeric ID)
SkeTermi = bwmorph(AllFragments,'endpoints'); % detect skeleton endpoints
[x, y] = find(SkeTermi==1);
all_tips = [x y];   % coordinates of all tips pixels in the skeleton

% associate each tip with its fragment label [row, column, fragmentID]
all_tips(:,3) = L(sub2ind(size(L),all_tips(:,1),all_tips(:,2)));

% may register tip orientation using parallel computing below
RegR = 2*R; % to register the direction, we need to consider only a local region around a tip

% save data\RegR.mat RegR;
save(fullfile(path_data,"RegR.mat"),"RegR")

tempInfo = zeros(size(all_tips,1),3);

% MultiCoreList = 1; % None 
MultiCoreList = 2; % Half
% MultiCoreList = 0; % Max

MultiCore = MultiCoreList;

tic;
if MultiCore==1
    h = waitbar(0,'Registering Tips ...');
    % loop through each tip
    for i = 1:size(all_tips,1)
        waitbar(i/size(all_tips,1),h);
        % calculate the orientation/direction of a tip with respect to its
        % centroid
        tempInfo(i,:) = TipReg(LL, all_tips(i,3), all_tips(i,1:2), RegR); % [orientation, row_cetroid, col_centroid]
    end
    close(h);
else
    delete(gcp);
    if MultiCore==2
        parpool ('local',round(feature('numCores')/2));
    else
        parpool ('local',feature('numCores'));
    end
    parfor i = 1:size(all_tips,1)
        tempInfo(i,:) = TipReg(LL, all_tips(i,3), all_tips(i,1:2), RegR)
    end
    delete(gcp);
end
toc;

% -------------------------------------------------------------------------
all_tips = [all_tips, tempInfo]; 
% [row, column, fragmentID, orientation, row_cetroid, col_centroid]
% -------------------------------------------------------------------------
% may register tip orientation using parallel computing above

if enable_preview == 1
    close(figure(1));
    figure('name','All Tips Detected (Orientations of 500 Tips Have Been Shown)');
    imshow(mat2gray(AllFragments));hold on; axis off;
    plot(all_tips(:,2),all_tips(:,1),'r+');
    for i = 1:100
        idx = ceil(rand * size(all_tips,1));
        text(all_tips(idx,2),all_tips(idx,1),[num2str(all_tips(idx,4))],'color','g');
    end
end

% save data\all_tips.mat all_tips;
save(fullfile(path_data,"all_tips.mat"),"all_tips")

disp('### Done.')

%% Grouping and analysis

disp('### Grouping and analysis parameter setting ...')
EditPixelSize = 0.02; % um, pixel size
MaxCurEdit = 1; % rad/um, max curvature
% this parameter is only used to help automatically set other parameters.
% If the user doesn't know the max curvature of your filament, you can
% mannually set other parameters.

% automatic set conditions
pixelsize = EditPixelSize;
MaxCur = MaxCurEdit;
FanRadius = 1/MaxCur/pixelsize;
FanAngle= 360/2/pi;

EditFanAngle=FanAngle;
C1=FanAngle;
C3=FanAngle/2;
FanR=FanRadius;
ShortFilamentEdit=FanRadius;

disp('### Done.')

%% preview and search criteria
% the seaRch angle and radius can be automatically set as above
% preview the search region and check whether it is suitable to cover most
% gaps that should be filled.

if enable_preview == 1
    disp('### Preview and search criteria ...')
    
    warning off;
    % load data\AllFragments;
    % load data\all_tips;
    
    close(figure(1));
    figure('name','Image of All Filamentous Fragments');
    imshow(mat2gray(AllFragments));axis off;
    title('Please double-click the tip where you want to check the region for searching');
    
    [Y1, X1]= ginput(1); % get user-selected point (ginput reverses the usual image order)
    % find the nearest tip
    k = dsearchn(all_tips(:,1:2),[X1 Y1]);
    k = k(1);
    BasePt = all_tips(k,:);
    
    % create the fan-shaped search region
    % FanR = FanR;
    FanAngle = EditFanAngle;
    FanEdge = FanPreview(FanAngle, AllFragments, FanR, BasePt(4), BasePt(1:2));
    
    % display the fan region
    close(figure(1));
    figure('name','Check the Region for Searching Partners');
    imshow(mat2gray(AllFragments));hold on; axis off;
    plot(FanEdge(:,2), FanEdge(:,1), 'g', 'LineWidth', 2);
    
    disp('### Done.')
end

%% Tip pairing and grouping
% To connect fragments skeleton segments into longer, continuous filaments.

% This algorihtm allow th case that a fragment is combined into mor than
% one composite filament in dependent on the maximum number of pairings it
% can form with other endpoints.

%% TIP PAIRING
% Each tip has a fan-shaped search area and a direction vector. If another
% tip lies within that fan and its orientation is compatitble, a connection
% is made. This creates a graph of possible fragment conncetions, which
% later grouping algorithms can trverse to reconstruct entire filament.

disp('### Tip pairing ...')

close(figure(1));

% load data\all_tips.mat; % all fragments endpoints with coordinates and orientations.
% load data\AllFragments; % the skeleton image
% load data\L.mat;        % the labeled skeleton image

% FanR = FanR;
FanAngle = EditFanAngle;
MIN_Angle_Diff = C1;    % minimum allowable angle difference between two tip directions.
MIN_Dist = FanR;        % maximum distance (search radius)
ShortFilamentEdit=MIN_Dist; 
MIN_GapAngle_Diff = C3; % maximum allowable gap angle difference
MIN_Info = [MIN_Angle_Diff,MIN_Dist,MIN_GapAngle_Diff];

label_list = all_tips(:,3); % will be updated; help to find another tip of non-first filament during searching

L_GlobalIndex = zeros(size(L)); % [Ny, Nx], L is an image of labeled fragments
all_tips(:,7) = (1:size(all_tips,1))'; % give each tip a global index
L_GlobalIndex(sub2ind(size(L_GlobalIndex),all_tips(:,1),all_tips(:,2))) = all_tips(:,7); % label the global id of tips in image
% go through all tips and register all searched information

MaxCur = MaxCurEdit;

% memory allocation
% each row corresponds to one tip
% each column will hold an index to a partner tip (up to 100)
new_partner_list = zeros(size(all_tips,1),100); % reserve the memory; a maximum of 100 partner tips allowed

% weights for optimal calculation (weights used to combine multiple pairing
% criteria into a single matching score)
C1weightEdit = 1; % weight for orientation difference degree
C3weightEdit = 1; % weight for gap orientation degree

C1weight = C1weightEdit;
C3weight = C3weightEdit;

% MultiCoreList = 1; % None
MultiCoreList = 2; % Half
% MultiCoreList = 3; % Full

MultiCore = MultiCoreList;
tic;
if MultiCore==1
    h = waitbar(0,'Tip Searching in Progress ...');
    for i = 1:size(all_tips,1)
        waitbar(i/size(all_tips,1),h);
        new_partner_list(i,:) = local_search( ...
            L_GlobalIndex, ...            % image of skeleton and its tips are labeled with global index
            label_list, ...               % this list will be frequently updated (if a tip has been used, it will be removed from this list)
            FanAngle, ...                 % size of the fan-shape searching region
            all_tips, ...                 % information of all tips
            FanR,  ...                    % this radius defines the range for searching
            all_tips(i,4), ...            % orientation of that current tip
            all_tips(i,1:2), ...          % coordinate of current tip
            MIN_Info, ...                 % Minimum conditions that should satisfy
            C1weight, ...                 % weight of croterion 1
            C3weight);                    % weight of croterion 3
        
    end
    close(h);
else
    delete(gcp);
    if MultiCore==2
        parpool ('local',round(feature('numCores')/2));
    else
        parpool ('local',feature('numCores'));
    end
    % parellel computing is used and it may take a few minutes for large data set
    parfor i = 1:size(all_tips,1)
        new_partner_list(i,:) = local_search( ...
            L_GlobalIndex, ...            % image of skeleton and its tips are labeled with global index
            label_list, ...               % this list will be frequently updated (if a tip has been used, it will be removed from this list)
            FanAngle, ...                 % size of the fan-shape searching region
            all_tips, ...                 % information of all tips
            FanR,  ...                    % this radius defines the range for searching
            all_tips(i,4), ...            % orientation of that current tip
            all_tips(i,1:2), ...          % coordinate of current tip
            MIN_Info, ...                 % Minimum conditions that should satisfy
            C1weight, ...                 % weight of croterion 1
            C3weight);                    % weight of croterion 3
        
    end
    delete(gcp);
end
toc;

% remove unused columns beyond the last filled partner index
for i = 1:size(new_partner_list,2)
    if sum(new_partner_list(:,i))==0 % all the values in this column is 0
        new_partner_list(:,i:end) = [];
        break;
    end
end

% save data\new_partner_list.mat new_partner_list;
save(fullfile(path_data,"new_partner_list.mat"),"new_partner_list")

all_tips(:,7) = 1:size(all_tips,1);         % add global index to all tips
all_tips(:,8) = ones(size(all_tips,1),1);   % this reserves to indicates number of lives

OverlapList = 1;    % None (for intricate network)
% OverlapList = 2;    % Allowed
% It is suggested to use the first option for network with high compleity.

Overlap = OverlapList;

h = waitbar(0,'Add Number of Lives for Each Filament ...');
% add the number of lives & partner tips to all tips
all_tips(:,9:end) = [];     % clear previous record
for i = 1:size(all_tips,1)  % for each tip
    waitbar(i/size(all_tips,1),h);
    temp_partner = new_partner_list(i,:);
    temp_partner(find(temp_partner==0)) = [];
    if ~isempty(temp_partner)
        all_tips(i,8) = length(temp_partner);                    % number of lives (# of partners)
        all_tips(i,9:(8 + length(temp_partner))) = temp_partner; % partner tip indices
    end
end
close(h);

% -------------------------------------------------------------------------
% all_tips : [row, col, fragment label, orientation, center-raw,
% center-col, global index, num of lives (partners), partner tip indices]
% -------------------------------------------------------------------------

% rebuild the labeled map that marks all tip positions by their global
% index.
% add the number of lives & partner tips to all tips
[L, num] = bwlabel(AllFragments,8);
L_GlobalIndex = zeros(size(L));
L_GlobalIndex(sub2ind(size(L),all_tips(:,1),all_tips(:,2))) = 1:length(all_tips(:,1));

% save data\L_GlobalIndex.mat L_GlobalIndex;
save(fullfile(path_data,"L_GlobalIndex.mat"),"L_GlobalIndex")

disp('Tip Search Done ! Please Proceed to GROUPING !')

% assign number of lives to each fragments according to the max number of lives of its two tips
for i = 1:num % for each fragment
    pair_index = find(all_tips(:,3)==i);
    if isempty(pair_index)
        continue;
    end
    all_tips(pair_index(1),8) = max(all_tips(pair_index,8)); % maximum number of connections either tip can form
    all_tips(pair_index(2),8) = max(all_tips(pair_index,8));
end
% this sets both tips' lives equal to the maximum number of connections
% either tip can form (i.e., how many composite filaments they can
% contribute to).

all_tips = biDirPairing(all_tips); % ensure that id tip A lists tip B as a partner, then B also lists A.

if Overlap == 1
    % if overlap is not allowed, each tip can participate in only on
    % connection.
    all_tips(:,8) = ones(size(all_tips,1),1); % if overlap is not allowed, the number of lives should be '1'
end

% save data\all_tips.mat all_tips;
save(fullfile(path_data,"all_tips.mat"),"all_tips")

% ****** structure of all_tips so far ******   #: Number
%           colume1            colume2           colume3         colume4         colume5            colume6        colume7          colume8          colume9               colume10                colume11          ...
% tip1     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip2     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip3     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip4     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% ......
% ****** structure of all_tips so far ******   #: Number
% assign number of lives to each fragments according to the max number of lives of its two tips

disp('### Done.')

%% GROUPING
% generate composite filaments
% filamentous fragment grouping starts

disp('### Grouping ...')

close(figure(1));

% load data\L.mat;
% load data\all_tips.mat;

NofLives = all_tips(:,8);
NofLives(find(NofLives==0)) = 1;
all_tips(:,8) = NofLives;
% assign number of lives to each fragments according to the max number of 
% lives of its two tips (holds how many tiems each tip can sitll be used in
% grouping)

new_partner_list = all_tips(:,9:end); % update the list of partners

all_filament = [];
all_connects = [];
[x, y] = find(L~=0);
Lpts = [x, y, L(sub2ind(size(L),x,y))]; % a list of all skeleton pixels, [row, col, fragment label]

h = waitbar(0,'Grouping in Progress');
tic;
for i = 1:num % for each fragment
    waitbar(i/num,h);
    L_list = [];

    label_list = all_tips(:,3); % labels (corresponding to L) of all fragments the global index of which corresponds to all_tips
    % get all the pixel coordinates of current fragment
    xx = Lpts(find(Lpts(:,3)==i),1);
    yy = Lpts(find(Lpts(:,3)==i),2); % pixels of fragment i

    if ~isempty(xx) % if this fragment still exists
        tips_index = find(all_tips(:,3)==i);                % find global index (the two tips of this fragment)
        newSinglefilament = [];                             % used to store a single filament
        newSinglefilament = [newSinglefilament; [xx yy]];   % add up to the single filament
        L_list = [L_list  i]; % list of labels used in this filament

        tip_dir1 = tips_index(1); % change to the first tip
        tip_dir2 = tips_index(2);
        connect_tips = all_tips(tip_dir1,1:2); % the design of this connect_tips is for quick filling between gaps later
        
        if NofLives(tip_dir1)<2
            % if number of lives < 1, mark fragment as 'used up'
            Lpts(find(Lpts(:,3)==i),3)=0;
            label_list(tips_index(1)) = 0;
            label_list(tips_index(2)) = 0;
            new_partner_list(find(new_partner_list==tip_dir1)) = 0;
            new_partner_list(find(new_partner_list==tip_dir2)) = 0;
        else
            % this fragment can still appear in multiple filaments
            NofLives(tips_index(1)) = NofLives(tips_index(1)) - 1; % reduce its number of lives by 1 if it is used
            NofLives(tips_index(2)) = NofLives(tips_index(2)) - 1; % reduce its number of lives by 1 if it is used
        end

        while 1+1==2
            can_index = new_partner_list(tip_dir1,:);   % get global indices of possible partner tips
            can_index(find(can_index==0)) = [];         % remove zeros

            if isempty(can_index) || ~isempty(find(L_list==all_tips(can_index(1),3)))  % if the current tip has no possible partner tip
                break;
            else
                optimal_index = can_index(1); % best-scoring partner

                % add new fragment to the current single filament
                xx = Lpts(find(Lpts(:,3)==label_list(optimal_index)),1);
                yy = Lpts(find(Lpts(:,3)==label_list(optimal_index)),2);
                newSinglefilament = [newSinglefilament; [xx yy]]; 

                L_list = [L_list  label_list(optimal_index)];
                connect_tips = [connect_tips;all_tips(optimal_index,1:2)];

                ll = NofLives(optimal_index);

                if NofLives(optimal_index)<2
                    Lpts(find(Lpts(:,3)==label_list(optimal_index)),3)=0;
                    new_partner_list(find(new_partner_list==tip_dir1)) = 0;
                    label_list(optimal_index) = 0; % remove this label but leave the other one
                    new_partner_list(find(new_partner_list==optimal_index)) = 0;
                else
                    NofLives(find(label_list==all_tips(optimal_index,3))) = NofLives(find(label_list==all_tips(optimal_index,3))) - 1; % reduce its number of lives by 1 if it is used
                end

                new_partner_list(tip_dir1,  find(new_partner_list(tip_dir1,:)==optimal_index)  ) = 0; % remove the tip from the current list of partner tips
                current_label = all_tips(optimal_index,3); % get the current label
                tip_dir1 = find(label_list==current_label); % find the other tip since only one left now
                if length(tip_dir1)==2
                    tip_dir1 = tip_dir1(find(tip_dir1~=optimal_index)); % find the other end of the fragment
                end

                if ll<2
                    new_partner_list(find(new_partner_list==tip_dir1)) = 0;
                    label_list(tip_dir1) = 0;
                end
                connect_tips = [connect_tips;all_tips(tip_dir1,1:2)];
            end
        end

        % gown from the other tip direction (backward)
        tip_dir2 = tips_index(2);
        connect_tips = [all_tips(tip_dir2,1:2);connect_tips];

        while 1+1==2
            can_index = new_partner_list(tip_dir2,:); % get global indices of possible partner tips
            can_index(find(can_index==0)) = []; % remove zeros

            if isempty(can_index) || ~isempty(find(L_list==all_tips(can_index(1),3)))% if the current tip has no possible partner tip
                break;
            else
                optimal_index = can_index(1);
                xx = Lpts(find(Lpts(:,3)==label_list(optimal_index)),1);
                yy = Lpts(find(Lpts(:,3)==label_list(optimal_index)),2);
                newSinglefilament = [newSinglefilament; [xx yy]]; % add new fragment to the current single filament
                L_list = [L_list  label_list(optimal_index)];
                connect_tips = [all_tips(optimal_index,1:2);connect_tips];

                ll = NofLives(optimal_index);
                if NofLives(optimal_index)<2
                    Lpts(find(Lpts(:,3)==label_list(optimal_index)),3)=0;
                    new_partner_list(find(new_partner_list==tip_dir2)) = 0;
                    label_list(optimal_index) = 0; % remove this label but leave the other one
                    new_partner_list(find(new_partner_list==optimal_index)) = 0;
                else
                    NofLives(find(label_list==all_tips(optimal_index,3))) = NofLives(find(label_list==all_tips(optimal_index,3))) - 1; % reduce its number of lives by 1 if it is used
                end

                new_partner_list(tip_dir2,  find(new_partner_list(tip_dir2,:)==optimal_index)  ) = 0; % remove the tip from the current list of partner tips
                current_label = all_tips(optimal_index,3); % get the current label
                tip_dir2 = find(label_list==current_label); % find the other tip since only one left now
                if length(tip_dir2)==2
                    tip_dir2 = tip_dir2(find(tip_dir2~=optimal_index)); % find to other end of the fragment
                end

                if ll<2
                    new_partner_list(find(new_partner_list==tip_dir2)) = 0;
                    label_list(tip_dir2) = 0;
                end
                connect_tips = [all_tips(tip_dir2,1:2);connect_tips];
            end
        end
        
        if isempty(all_filament) && isempty(all_connects)
            all_filament(1:size(newSinglefilament,1),1:2,1) = newSinglefilament;
            all_connects(1:size(connect_tips,1),1:2,1) = connect_tips;
        else
            all_filament(1:size(newSinglefilament,1),1:2,size(all_filament,3)+1) = newSinglefilament;
            all_connects(1:size(connect_tips,1),1:2,size(all_connects,3)+1) = connect_tips;
        end
    end
end
toc;

% -------------------------------------------------------------------------
% all_filament
% [max_number_of_pixels_in_any_filament, [row,col], number_of_filaments (composite)]
% all_connect
% [max_number_of_tips+per_filament, [row,col], number_of_filaments (composite)]
% -------------------------------------------------------------------------

disp('Grouping Done !')
close(h);

% close(figure(1));

% filamentous fragment grouping ends
% save data\all_filament.mat all_filament;
% save data\all_connects.mat all_connects;
% all_connects_shortlist = all_connects;
% save data\all_connects_shortlist.mat all_connects_shortlist;

save(fullfile(path_data,"all_filament.mat"),"all_filament")
save(fullfile(path_data,"all_connects.mat"),"all_connects")
all_connects_shortlist = all_connects;
save(fullfile(path_data,"all_connects_shortlist.mat"),"all_connects_shortlist")

disp('### Done.')

%% Filament Sorting
% sort composite filaments
% sort each filament's pixels into meaningful order
% fill samll gaps and produces a contibuous curve
% compute analysis metrics for each filament

disp('### Filament Sorting ...')

close(figure(1));

% load data\all_filament.mat;
% load data\all_connects.mat;
% load data\L;

maskI = zeros(size(L));
Fullength = 5*size(all_filament,1); % maximum possible sorted length
all_sorted_filament = zeros(Fullength,2,size(all_filament,3));

MultiCore = MultiCoreList;

tic;
if MultiCore==1
    h = waitbar(0,'Analysis in Progress ...');
    for i = 1:size(all_filament,3)
        waitbar(i/size(all_filament,3),h);
        [all_sorted_filament(:,:,i), AnalysisInfo(i,:)]= SortFilament( ...
            all_filament(:,:,i), maskI, all_connects(:,:,i), Fullength);
    end
    close(h);
else
    delete(gcp);
    if MultiCore==2
        parpool ('local',round(feature('numCores')/2));
    else
        parpool ('local',feature('numCores'));
    end
    parfor i = 1:size(all_filament,3)
        [all_sorted_filament(:,:,i), AnalysisInfo(i,:)]= SortFilament( ...
            all_filament(:,:,i), maskI, all_connects(:,:,i), Fullength);
    end
    delete(gcp);
end
toc;

% remove empty rows from the bottom
for i = 1:size(all_sorted_filament,1)
    temp = all_sorted_filament(i,:,:);
    if(sum(temp(:)))==0
        all_sorted_filament(i:end,:,:) = [];
        break;
    end
end

disp('Analysis and Sorting Done !');

% save data\all_sorted_filament.mat all_sorted_filament;
% save data\AnalysisInfo.mat AnalysisInfo;
save(fullfile(path_data,"all_sorted_filament.mat"),"all_sorted_filament")
save(fullfile(path_data,"AnalysisInfo.mat"),"AnalysisInfo")
% [filament orientation (degrees), total path length (sum of pixel
% distances), end-to-end distance, center-row, center-col]

disp('### Done.')

%% Remove short filaments
% remove ungrouped filament fragments.
disp('### Removing short filament ...')

close(figure(1));
close(figure(2));

% load data\AnalysisInfo;
% load data\all_sorted_filament;
% load data\AllFragments.mat;
% load data\all_connects.mat;

ShortFilamentEdit =20; % pixel
RemoveUngrp = 1; % whether to remove ungrounded

% remove short filaments
ShortFilament= ShortFilamentEdit;
RemoveIdx = [];
h = waitbar(0,'Removing Short Filaments...');
for i = 1:size(all_sorted_filament,3)
    waitbar(i/size(all_sorted_filament,3),h);
    if AnalysisInfo(i,2) <= ShortFilament
        RemoveIdx = [RemoveIdx i];
    end
end

close(h);
all_sorted_filament(:,:,RemoveIdx) = [];
AnalysisInfo(RemoveIdx,:) = [];
all_connects(:,:,RemoveIdx) = [];

% remove ungrouped below
% RemoveUngrp = RemoveUngrp;
if RemoveUngrp==1
    % AllFragments = im2bw(AllFragments);
    RemoveIdx = [];

    h = waitbar(0,'Removing Ungrouped Fragments...');
    for i = 1:size(all_sorted_filament,3)
        waitbar(i/size(all_sorted_filament,3),h);
        temp = all_sorted_filament(:,:,i);
        temp(find(temp(:,1)==0),:) = [];
        % check if the filament pixels come entirely from the original
        % skeleton
        if size(temp,1)==sum(AllFragments(sub2ind(size(AllFragments),temp(:,1),temp(:,2))))
            RemoveIdx = [RemoveIdx  i];
        end
    end
    close(h);

    all_sorted_filament(:,:,RemoveIdx) = [];
    AnalysisInfo(RemoveIdx,:) = [];
    all_connects(:,:,RemoveIdx) = [];
end
% remove ungrouped above

all_connects_shortlist = all_connects;
% save data\AnalysisInfo.mat AnalysisInfo;
% save data\all_sorted_filament.mat all_sorted_filament;
% save data\all_connects_shortlist.mat all_connects_shortlist;

save(fullfile(path_data,"AnalysisInfo.mat"),"AnalysisInfo")
save(fullfile(path_data,"all_sorted_filament.mat"),"all_sorted_filament")
save(fullfile(path_data,"all_connects_shortlist.mat"),"all_connects_shortlist")

disp('### Done.')

%% Optional: Goto GUI for Manual Correction
% disp('### Manual correction ...')
% FanR = str2num(get(handles.FanR,'String'));
% % save data\FanR.mat FanR;
% save(fullfile(path_data,"FanR.mat"),'FanR')
% ManCorr;
% disp('### Done.')

%% Dispaly Results
% SIFNE provides the following analysis
% - all detected filaments
% - junctions
% - histogram of orientations
% - curvature
% - export into excel

%% Detected filament
% Image of the skeleton of binarized image overlaid with composite
% filaments shown in different colors. Cell boundary is indicated in white.

disp('### Display results: detected filament ...')

warning off;
close(figure(1));close(figure(2));

% load data\R.mat;
% load data\AllFragments.mat;
% load data\L.mat;
% load data\all_sorted_filament.mat all_sorted_filament;
% load data\AnalysisInfo.mat;
% load data\ROI_Mask;

% reconstruct network
allpts_IncludeDup = [];
Overlay_Map = zeros(size(L)); % how many times each pixel belongs to any filament.

h = waitbar(0,'Retrieve All Filaments');
for i = 1:size(all_sorted_filament,3) % for each filament
    waitbar(i/size(all_sorted_filament,3),h);
    temp1 = all_sorted_filament(:,:,i); % get all coordinates
    temp1(find(temp1(:,1)==0),:) = []; % remove zeros
    allpts_IncludeDup = [allpts_IncludeDup;temp1];
    % check points used more than once
    Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) = Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) + 1;
end
close(h);

% display
figure(1);
imshow(mat2gray(AllFragments( ...
    (R+1):(size(AllFragments,1)-R), ...
    (R+1):(size(AllFragments,2)-R))));
hold on;axis off;
% screen_size = get(0, 'ScreenSize'); 
% set(figure(1), 'Position', [0 0 screen_size(3) screen_size(4) ] );

ColorList = rand(32,3);
temp = 1:size(all_sorted_filament,3);

for i = 1:32
    idx = find(mod(temp,32)==(i-1));
    temp1 = all_sorted_filament(:,1,idx);  temp1 = temp1(:);  temp1(find(temp1==0)) = [];
    temp2 = all_sorted_filament(:,2,idx);  temp2 = temp2(:);  temp2(find(temp2==0)) = [];
    plot(temp2-R,temp1-R,'.','color',ColorList(i,:),'MarkerSize',6);hold on;
end

Overlap = OverlapList;
if Overlap==2
    [x, y] = find(Overlay_Map>1);
    plot(y-R,x-R,'r.','MarkerSize',6);hold on;
end

% show cell boundary
B = bwboundaries(ROI_Mask);
B = B{1};
plot(B(:,2),B(:,1),'.','color',[1 1 1],'MarkerSize',6);hold on;

% mkdir result;
% saveas(figure(1),'result\Extracted_Filaments.fig');
saveas(figure(1),fullfile(path_result,'Extracted_Filaments.fig'))

disp('### Done.')

%% Junctions
% Enlarged image of all composite filaments (black) overlaid with all
% centroids of junctions (green). The background image is distance map as a
% function of the distance to cell edge. Unit of color bar: um.
% Distribution of junctions as a function fo their distance to cell
% boundary.

disp('### Display results: Junctions ...')

warning off;
% close(figure(1));close(figure(2));

% load data\R.mat;
% load data\OriginImg.mat;
% load data\L.mat;
% load data\all_sorted_filament.mat all_sorted_filament;
% load data\ROI_Mask;

PixelSize =EditPixelSize;

Overlay_Map = zeros(size(L));

h = waitbar(0,'Checking Junctions ...');
% counting junction pixel
for i = 1:size(all_sorted_filament,3)
    waitbar(i/size(all_sorted_filament,3),h);
    temp1 = all_sorted_filament(:,:,i);
    temp1(find(temp1(:,1)==0),:) = [];
    Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) = Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) + 1;
end
close(h);
[x, y] = find(Overlay_Map>1);

% collapse nearby junction pixels into single points
CroMap = zeros(size(L));
CroMap(sub2ind(size(L),x,y)) = 1;
for i = 1:length(x)
    temp = CroMap((x(i)-1):(x(i)+1), (y(i)-1):(y(i)+1));
    if sum(temp(:))>1
        CroMap(x(i),y(i)) = 0;
    end
end
[x, y] = find(CroMap==1);
NewCrPts = [x y]; % final junction coordinates

% save data\NewCrPts.mat NewCrPts;
save(fullfile(path_data,"NewCrPts.mat"),"NewCrPts")

disF=bwdist(bwmorph(ROI_Mask,'remove')); % compute distance to the boundary
mask = ROI_Mask;
mask = single(mask);
mask(mask==0)=0;
disF = disF.*mask; % set distance outside ROI to 0
disF = disF*PixelSize;

% dispaly
figure(2);
imagesc(disF);colormap(jet);hold on;axis off;axis image;colorbar;
[xx, yy] = find(Overlay_Map~=0);
plot(yy,xx,'k.');
plot(y,x,'g.','MarkerSize',15);title('Distribution of Junctions');
B = bwboundaries(ROI_Mask);
B = B{1};
plot(B(:,2),B(:,1),'.','color',[1 1 1],'MarkerSize',6);hold on;

disF=bwdist(bwmorph(ROI_Mask,'remove'));
mask = ROI_Mask;
mask = single(mask);
mask(mask==0)=-1;
disF = disF.*mask;
d = disF(sub2ind(size(disF),x,y))*PixelSize;

figure(3);
%         histogram(d);%axis square;
histfit(d,50,'kernel');xlim([0 inf]);
ylabel('Frequency');
xlabel('Distance to Cell Edge (\mum)');
title('Distribution of Junctions');

% mkdir result;
% saveas(figure(1),'result\Distribution_Junctions.fig');
% saveas(figure(2),'result\Distribution_Junctions_Analysis.fig');
saveas(figure(2),fullfile(path_result,'Distribution_Junctions.fig'));
saveas(figure(3),fullfile(path_result,'Distribution_Junctions_Analysis.fig'));

disp('### Done.')


%% Fialment Orientation
% Rose plot of all filament orientations. The orientations of filaments
% ranges form -90 to 90 degrees.
% Spatial distribution of all orientations as a funciton fo the distance
% between filaments centroids and cell edge. Unit of colorbar:
% counts/frequency.

disp('### Display results: filament orientation ...')

warning off;
close(figure(1));close(figure(2));

% load data\all_sorted_filament.mat;
% load data\AnalysisInfo.mat;
% load data\L;
% load data\ROI_Mask;

% plot of information of all filaments
PixelSize = EditPixelSize;

figure(1);
h=rose(pi*AnalysisInfo(:,1)/180,30);axis off;
set(h,'linewidth',3)
axis square;
title('Histogram of Filaments Orientations');

% mkdir result;
% saveas(figure(1),'result\Histogram_of_Orientations.fig');
saveas(figure(1),fullfile(path_result,'Histogram_of_Orientations.fig'));

allc = AnalysisInfo(:,4:5);
disF=bwdist(bwmorph(ROI_Mask,'remove'));
mask = ROI_Mask;
mask = single(mask);
mask(mask==0)=-1;
disF = disF.*mask;
d = disF(sub2ind(size(L),ceil(allc(:,1)),ceil(allc(:,2))))*PixelSize;

finalmap = zeros(181,ceil(max(d)));

h = waitbar(0,'generating colormap ...');
for i = 1:length(d)
    waitbar(i/size(all_sorted_filament,3),h);
    if ceil(d(i))>0
        finalmap(ceil(AnalysisInfo(i,1)+91), ceil(d(i))) = finalmap(ceil(AnalysisInfo(i,1)+91), ceil(d(i))) + 1;
    end
end
close(h);

figure(2);
imagesc(finalmap);colormap(jet);colorbar;
set(gca,'ytick',[]);
xlabel('Distance to Cell Edge (\mum)');
ylabel('-90 degrees to 90 degrees');
title('Distribution of Filament Orientations');

% saveas(figure(2),'result\Distribution_Orientations.fig');
saveas(figure(2),fullfile('Distribution_Orientations.fig'));

disp('### Done.')

%% Curvature
% Enlarged image of filament curvatures. Unit of color bar: um-1
% Histogram of curvatures all composite filament pixels as a function of
% the distance between the centroids of filaments and cell edge.

disp('### Display results: Curvature ...')

warning off;
close(figure(1));close(figure(2));

% load data\all_sorted_filament;
% load data\ROI_Mask;
% load data\AllFragments;

curR = round(FanR/2);
PixelSize = EditPixelSize;

disF=bwdist(bwmorph(ROI_Mask,'remove'));
mask = ROI_Mask;
mask = single(mask);
mask(mask==0)=-1;
disF = disF.*mask;

allcur = [];
alldSDF = [];
M = zeros(size(AllFragments));

h = waitbar(0,'Calculating Curvatures ...');
all_filament_curs = [];
all_mean_cur = zeros(1,size(all_sorted_filament,3));
for i = 1:size(all_sorted_filament,3)
    waitbar(i/size(all_sorted_filament,3),h);
    F = all_sorted_filament(:,:,i);
    F(find(F(:,1)==0),:) = [];
    Dlist = 0;
    for j = 2:size(F,1)
        Dlist = [Dlist  Dlist(j-1)+pdist([F(j-1,:); F(j,:)])];
    end
    NoCurlist = unique([find(Dlist<=curR)  find(Dlist>=(Dlist(end)-curR))]);
    if length(NoCurlist)~=size(F,1)
        for j = 1:size(F,1)
            if isempty(find(NoCurlist==j))
                x2 = F(j,1);
                y2 = F(j,2);
                D = Dlist - Dlist(j);
                D1 = abs(D - curR);
                idx1 = find(D1==min(D1));
                D2 = abs(D - (-curR));
                idx2 = find(D2==min(D2));
                x1 = F(idx1,1);
                y1 = F(idx1,2);
                x3 = F(idx2,1);
                y3 = F(idx2,2);
                alldSDF = [alldSDF  disF(sub2ind(size(disF),x2,y2))*PixelSize];
                cur = 2*abs((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1)) ./sqrt(((x2-x1).^2+(y2-y1).^2)*((x3-x1).^2+(y3-y1).^2)*((x3-x2).^2+(y3-y2).^2));
                allcur = [allcur  cur/PixelSize];
                M(F(j,1),F(j,2)) = cur;
                all_filament_curs(j,1,i) = x2;
                all_filament_curs(j,2,i) = y2;
                all_filament_curs(j,3,i) = cur/PixelSize;
                all_filament_curs(j,4,i) = 1;% flag for calculation done for this point
            end
        end
        k = find(all_filament_curs(:,4,i)==1);
        all_mean_cur(i) = mean(all_filament_curs(k,3,i));
    end
end
close(h);
M = M/PixelSize;
figure(1);imagesc(M); colormap(jet);colorbar; 
title('Distribution of Curvatures');axis off;axis image;
  
figure(2);
subplot(1,2,1);
histfit(all_mean_cur,50,'kernel');xlim([0 inf]);
xlabel('Curvature (unit: \mum^-^1)');
ylabel('Frequency');
title('Distribution of Means of Filament Curvatures');

All_Curs = [];
All_Curs_4plot = [];
disF=bwdist(bwmorph(ROI_Mask,'remove'));
mask = ROI_Mask;
mask = single(mask);
mask(mask==0)=-1;
disF = disF.*mask;
h = waitbar(0,'Calculating the Distribution of Curvatures ...');
for m = 1:size(all_filament_curs,3)
    waitbar(m/size(all_filament_curs,3),h);
    Cur = all_filament_curs(:,:,m);
    F = all_sorted_filament;
    F(find(F(:,1)==0),:) = [];
    Cur(find(Cur(:,4)==0),:) = [];
    if isempty(Cur)
        All_Curs_4plot = [All_Curs_4plot;  disF(sub2ind(size(disF),round((F(1,1)+F(end,1))/2),round((F(end,2)+F(1,2))/2)))   0];
        All_Curs = [All_Curs, 0];
    else
        All_Curs_4plot = [All_Curs_4plot;  disF(sub2ind(size(disF),Cur(:,1),Cur(:,2)))   Cur(:,3)];
        All_Curs = [All_Curs, Cur(:,3)'];
    end
end
close(h);
All_Curs_4plot(:,1) = round(All_Curs_4plot(:,1));
temp_mean_cur = [];
x1 = [];
y1 = [];
for i = min(All_Curs_4plot(:,1)): max(All_Curs_4plot(:,1))
    k = find(All_Curs_4plot(:,1)==i);
    temp_mean_cur = [temp_mean_cur  mean(All_Curs_4plot(k,2))];
    if length(k)>5
        x1 = [x1  i];
        y1 = [y1  mean(All_Curs_4plot(k,2))];
    end
end

subplot(1,2,2);
plot(x1*0.02, y1,  'r');hold on;axis([0 inf 0 inf]);
xlabel('Distance to Cell Edge (unit: \mum)');ylabel('Mean Curvature (unit: \mum^-^1)');
title('Distribution of Curvatures');

% save data\all_filament_curs.mat all_filament_curs;
save(fullfile(path_data,"all_filament_curs.mat"),"all_filament_curs")

% mkdir result;
% saveas(figure(1),'result\Distribution_of_Curvatures.fig');
% saveas(figure(2),'result\Distribution_of_Curvatures_Analysis.fig');
saveas(figure(1),fullfile(path_result,'Distribution_of_Curvatures.fig'));
saveas(figure(2),fullfile(path_result,'Distribution_of_Curvatures_Analysis.fig'));

disp('### Done.')

%% Export into Excel
% Export the information of composite filaments, junctions and fragment 
% linkage into an excel file for more customized analysis. 
% The exported excel file includes 4 worksheets as follows. 
% Worksheet 1: Information of all composite filaments
% Worksheet 2: Information of all filament fragments
% Worksheet 3: Linkage information before removing short filaments and 
% ungrouped fragments
% Worksheet 4: Linkage information after removing short filaments and 
% ungrouped fragments

disp('### Exporting into Excel ...')

warning off;
close(figure(1));close(figure(2));

% load data\AnalysisInfo;
% load data\all_sorted_filament;
% load data\NewCrPts;
% load data\Size_Junc;
% load data\ROI_Mask;
% load data\L.mat;
% load data\all_tips.mat;
% load data\all_connects.mat;
% load data\all_connects_shortlist.mat;

% generate information of all filaments
PixelSize = EditPixelSize;

titles = {'Filament ID','1st X pos','1st Y pos','last X pos','last Y pos',...
    'Orientation','Total Length','End-to-End Distance','Centroid X','Centroid Y'};
titles = [titles,repmat({'X','Y'},[1 size(all_sorted_filament,1)]) ];
InfoExcel = zeros(size(all_sorted_filament,3)*2, size(all_sorted_filament,1)*2+9);
% one row per filament

h = waitbar(0,'Generating information of composite filaments ...');
for i = 1:size(all_sorted_filament,3)
    waitbar(i/size(all_sorted_filament,3),h);

    curr_filament = all_sorted_filament(:,:,i);
    curr_filament(find(curr_filament(:,1)==0),:) = []; % remove padding
    InfoExcel((i*2-1), 1:9) = [curr_filament(1,:), curr_filament(end,:), AnalysisInfo(i,:)]; % start x, y, end x, y, metrics
    
    temp = zeros(1,size(all_sorted_filament,1)*2); % coordinate fo each pixel
    temp((1:size(all_sorted_filament,1))*2-1) = all_sorted_filament(:,1,i);
    temp((1:size(all_sorted_filament,1))*2) = all_sorted_filament(:,2,i);
    InfoExcel((i*2-1), 10:end) = temp;

    % save junctions belonging to this filament
    ks = dsearchn(curr_filament,NewCrPts);
    for j = 1:length(ks)
        if pdist([curr_filament(ks(j),:); NewCrPts(j,:)])<=Size_Junc
            InfoExcel((i*2), ((9+ks(j)*2-1):(9+ks(j)*2))) = NewCrPts(j,:); % coodinates of junctions are stored in the next row
        end
    end
end

FragID = 1:size(all_sorted_filament,3);
FragID = [FragID; FragID]; % one row for filament, onw row for junction
FragID = FragID(:);
InfoExcel = [FragID, InfoExcel];

% save data\InfoExcel.mat InfoExcel;
save(fullfile(path_data,"InfoExcel.mat"),"InfoExcel")

close(h);
InfoExcel = [titles;num2cell(InfoExcel)];

% ------ generate fragment information ------
FragmentInfo = zeros(num,size(L,1));
MultiCore = MultiCoreList;

if MultiCore==1
    h = waitbar(0,'Fragment Information ...');
    for i = 1:num
        waitbar(i/num,h);
        FragmentInfo(i,:) = GenFragmentInfo(L,i,all_tips,FragmentInfo(i,:));
    end
    close(h);
else
    delete(gcp);
    if MultiCore==2
        parpool ('local',round(feature('numCores')/2));
    else
        parpool ('local',feature('numCores'));
    end
    % parellel computing is used and it may take a few minutes for large data set

    % load data\L.mat;
    % L = L;
    % load data\all_tips.mat;
    % all_tips = all_tips;

    parfor i = 1:num
        FragmentInfo(i,:) = GenFragmentInfo(L,i,all_tips,FragmentInfo(i,:));
    end
end


sumlist = sum(FragmentInfo,1);
FragmentInfo(:,(find(sumlist==0))) = [];

% save data\FragmentInfo.mat FragmentInfo;
save(fullfile(path_data,"FragmentInfo.mat"),"FragmentInfo")

MaxFragL = size(FragmentInfo,2)-6;

disp('Fragment Information Generated !');
titles = {'Fragment ID','# of Pixels','Beginning X','Beginning Y','Ending X','Ending Y'};
titles = [titles,repmat({'X','Y'},[1 (size(FragmentInfo,2)-6)/2]) ];
FragmentInfo = [titles;num2cell(FragmentInfo)];

% ------ generate linkage information before removing short------

LinkInfo = zeros(size(all_connects,1),2+MaxFragL*2,size(all_connects,3));
if MultiCore==1
    h = waitbar(0,'Linkage1 Information ...');
    for i = 1:size(all_connects,3)
        waitbar(i/size(all_connects,3),h);
        LinkInfo(:,:,i) = GenLinkageInfo(i,L,all_connects(:,:,i),LinkInfo(:,:,i));
    end
    close(h);
else
    % load data\L.mat;
    % L = L;
    parfor i = 1:size(all_connects,3)
        LinkInfo(:,:,i) = GenLinkageInfo(i,L,all_connects(:,:,i),LinkInfo(:,:,i));
    end
end

disp('Linkage1 Information Generated !');


LinkageInfo1 = [];
h = waitbar(0,'Integrating Linkage1 Information ...');
tic;
for i = 1:size(LinkInfo,3)
    waitbar(i/size(LinkInfo,3),h);
    tempLinkage = LinkInfo(:,:,i);
    tempLinkage(find(tempLinkage(:,1)==0),:) = [];
    LinkageInfo1 = [LinkageInfo1; tempLinkage];
end
close(h);
disp('Linkage1 Integration Done !');
sumlist = sum(LinkageInfo1,1);
LinkageInfo1(find(LinkageInfo1(:,1)==0),:) = [];
LinkageInfo1(:,(find(sumlist==0))) = [];

% save data\LinkageInfo1.mat LinkageInfo1;
save(fullfile(path_data,"LinkageInfo1.mat"),"LinkageInfo1")

titles = {'Composite Filament ID','Fragment ID'};
titles = [titles,repmat({'X','Y'},[1 (size(LinkageInfo1,2)-2)/2]) ];
LinkageInfo1 = [titles;num2cell(LinkageInfo1)];

% -------------------------------------------------------------------------
% Composite Filament ID | Fragment ID | X1 | Y1 | X2 | Y2 | …
% -------------------------------------------------------------------------

% ------ generate linkage information after removing short------

LinkInfo = zeros(size(all_connects_shortlist,1),2+MaxFragL*2,size(all_connects_shortlist,3));
if MultiCore==1
    h = waitbar(0,'Linkage2 Information ...');
    for i = 1:size(all_connects_shortlist,3)
        waitbar(i/size(all_connects_shortlist,3),h);
        LinkInfo(:,:,i) = GenLinkageInfo(i,L,all_connects_shortlist(:,:,i),LinkInfo(:,:,i));
    end
    close(h);
else
    parfor i = 1:size(all_connects_shortlist,3)
        LinkInfo(:,:,i) = GenLinkageInfo(i,L,all_connects_shortlist(:,:,i),LinkInfo(:,:,i));
    end
end

disp('Linkage2 Information Generated !');


LinkageInfo2 = [];
h = waitbar(0,'Integrating Linkage2 Information ...');
tic;
for i = 1:size(LinkInfo,3)
    waitbar(i/size(LinkInfo,3),h);
    tempLinkage = LinkInfo(:,:,i);
    tempLinkage(find(tempLinkage(:,1)==0),:) = [];
    LinkageInfo2 = [LinkageInfo2; tempLinkage];
end
close(h);

disp('Linkage2 Integration Done !');
sumlist = sum(LinkageInfo2,1);
LinkageInfo2(find(LinkageInfo2(:,1)==0),:) = [];
LinkageInfo2(:,(find(sumlist==0))) = [];

% save data\LinkageInfo2.mat LinkageInfo2;
save(fullfile(path_data,"LinkageInfo2.mat"),"LinkageInfo2")

titles = {'Composite Filament ID','Fragment ID'};
titles = [titles,repmat({'X','Y'},[1 (size(LinkageInfo2,2)-2)/2]) ];
LinkageInfo2 = [titles;num2cell(LinkageInfo2)];

% -------------------------------------------------------------------------
% Composite Filament ID | Fragment ID | X1 | Y1 | X2 | Y2 | …
% -------------------------------------------------------------------------

% mkdir result;
% write inforamtion to excel
% cd result;
% xlswrite('IntegratedInfo.xlsx',InfoExcel,1,'A1');
% xlswrite('IntegratedInfo.xlsx',FragmentInfo,2,'A1');
% xlswrite('IntegratedInfo.xlsx',LinkageInfo1,3,'A1');
% xlswrite('IntegratedInfo.xlsx',LinkageInfo2,4,'A1');

path_xls = fullfile(path_result,'IntegratedInfo.xlsx');
xlswrite(path_xls,InfoExcel,1,'A1');
xlswrite(path_xls,FragmentInfo,2,'A1');
xlswrite(path_xls,LinkageInfo1,3,'A1');
xlswrite(path_xls,LinkageInfo2,4,'A1');

% curDir = pwd;
% filaname = [curDir,'\IntegratedInfo.xlsx'];
filaname = char(fullfile(path_result,'IntegratedInfo.xlsx'));

e = actxserver('Excel.Application');
ewb = e.Workbooks.Open(filaname);
ewb.Worksheets.Item(1).Name = 'Ultimate Filaments';
ewb.Worksheets.Item(2).Name = 'Fragment Info';
ewb.Worksheets.Item(3).Name = 'Linkage Info1';
ewb.Worksheets.Item(4).Name = 'Linkage Info2';
ewb.Save
ewb.Close(false);
e.Quit;

% cd ..;
% msgbox('Excel Generated !');

disp('### Done.')

%% Complete and Save settings

disp('### Complete and saving settings ...')

% FanR = FanR;
% EditFanAngle = EditFanAngle;
% C1 = C1;
C1weight = C1weightEdit;
C2 = FanR;
C3weight = C3weightEdit;
% C3 = C3;
% EditPixelSize = EditPixelSize;
% ShortFilamentEdit = ShortFilamentEdit;    
MaxCur = MaxCurEdit;

% save user settings
% fileID = fopen('UserSettings\GroupingSettings.txt','w');
fileID = fopen(fullfile(path_settings,'GroupingSettings.txt'),'w');
fprintf(fileID,['Radius of Searching Fan (pixels):              ',num2str(FanR),'\r\n']);
fprintf(fileID,['Angle of Searching Fan (degrees):              ',num2str(EditFanAngle),'\r\n']);
fprintf(fileID,['Criterion1 (Orientation Difference) (degrees): ',num2str(C1),'\r\n']);
fprintf(fileID,['Criterion1 Weight:                             ',num2str(C1weight),'\r\n']);
fprintf(fileID,['Criterion2 (Distance) (pixels):                ',num2str(C2),'\r\n']);
fprintf(fileID,['Criterion3 (Gap Orientation) (degrees):        ',num2str(C3),'\r\n']);
fprintf(fileID,['Criterion3 Weight:                             ',num2str(C3weight),'\r\n']);
fprintf(fileID,['Pixel Size (um):                               ',num2str(EditPixelSize),'\r\n']);
fprintf(fileID,['Short Fragments to Remove (pixels):            ',num2str(ShortFilamentEdit),'\r\n']);
fprintf(fileID,['Maximum Curvature (radian/um):                 ',num2str(MaxCur),'\r\n']);

% go to next user interface
close all;

disp('### Done.')
disp('### ALL DONE. ###')