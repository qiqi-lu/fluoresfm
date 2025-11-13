% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function varargout = SegmentB4Grouping(varargin)
% SEGMENTB4GROUPING MATLAB code for SegmentB4Grouping.fig
%      SEGMENTB4GROUPING, by itself, creates a new SEGMENTB4GROUPING or raises the existing
%      singleton*.
%
%      H = SEGMENTB4GROUPING returns the handle to a new SEGMENTB4GROUPING or the handle to
%      the existing singleton*.
%
%      SEGMENTB4GROUPING('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEGMENTB4GROUPING.M with the given input arguments.
%
%      SEGMENTB4GROUPING('Property','Value',...) creates a new SEGMENTB4GROUPING or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SegmentB4Grouping_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SegmentB4Grouping_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SegmentB4Grouping

% Last Modified by GUIDE v2.5 11-Nov-2025 15:46:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @SegmentB4Grouping_OpeningFcn, ...
    'gui_OutputFcn',  @SegmentB4Grouping_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SegmentB4Grouping is made visible.
function SegmentB4Grouping_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SegmentB4Grouping (see VARARGIN)

% Choose default command line output for SegmentB4Grouping
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SegmentB4Grouping wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SegmentB4Grouping_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



% --- Executes on button press in SetGrpParametersBtn.
function SetGrpParametersBtn_Callback(hObject, eventdata, handles)
% hObject    handle to SetGrpParametersBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ThreshOpt1Edit = str2num(get(handles.ThreshOpt1Edit,'String'));
SizofJuncEdit = str2num(get(handles.SizofJuncEdit,'String'));
MinNofPixels = str2num(get(handles.MinNofPixels,'String'));


% save user settings
fileID = fopen('UserSettings\SegmentationSettings.txt','w');
fprintf(fileID,['Segmentation Option1 Threshold value:      ',num2str(ThreshOpt1Edit),'\r\n']);
fprintf(fileID,['Size of Junctions to Remove (pixels):      ',num2str(SizofJuncEdit),'\r\n']);
fprintf(fileID,['Short Fragments to Remove (pixels):        ',num2str(MinNofPixels),'\r\n']);

% go to next user interface
close all;
GrpAndAnalysis;

% detect tip points





function MinNofPixels_Callback(hObject, eventdata, handles)
% hObject    handle to MinNofPixels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of MinNofPixels as text
%        str2double(get(hObject,'String')) returns contents of MinNofPixels as a double


% --- Executes during object creation, after setting all properties.
function MinNofPixels_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MinNofPixels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function EditSizeofReg_Callback(hObject, eventdata, handles)
% hObject    handle to EditSizeofReg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of EditSizeofReg as text
%        str2double(get(hObject,'String')) returns contents of EditSizeofReg as a double

% --- Executes on button press in RegTipsBtn.
function RegTipsBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RegTipsBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load data\AllFragments.mat;
load data\Allpts.mat;
load data\L.mat;
load data\R.mat;
LL = L;
SkeTermi = bwmorph(AllFragments,'endpoints');
[x y] = find(SkeTermi==1);
all_tips = [x y];

all_tips(:,3) = L(sub2ind(size(L),all_tips(:,1),all_tips(:,2)));

% may register tip orientation using parallel computing below
RegR = 2*R; % to register the direction, we need to consider only a local region around a tip
save data\RegR.mat RegR;
tempInfo = zeros(size(all_tips,1),3);

MultiCore = get(handles.MultiCoreList,'Value');
tic;
if MultiCore==1
    h = waitbar(0,'Registering Tips ...');
    for i = 1:size(all_tips,1)
        waitbar(i/size(all_tips,1),h);
        tempInfo(i,:) = TipReg(LL, all_tips(i,3), all_tips(i,1:2), RegR);
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
all_tips = [all_tips, tempInfo];
% may register tip orientation using parallel computing above

close(figure(1));
figure('name','All Tips Detected (Orientations of 500 Tips Have Been Shown)');
imshow(mat2gray(AllFragments));hold on; axis off;plot(all_tips(:,2),all_tips(:,1),'r+');
for i = 1:100
    idx = ceil(rand * size(all_tips,1));
    text(all_tips(idx,2),all_tips(idx,1),[num2str(all_tips(idx,4))],'color','g');
end
save data\all_tips.mat all_tips;


% --- Executes on button press in RemoveShortBtn.
function RemoveShortBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RemoveShortBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% remove very short fragment

load data\AllFragments.mat;
load data\R.mat
load data\OriginImg.mat
load data\Allpts.mat;
load data\L.mat;
MIN_FragmentLength = str2num(get(handles.MinNofPixels,'String'));
AllFragments = bwareaopen(AllFragments, MIN_FragmentLength);
[L num] = bwlabel(AllFragments,8);
[x y] = find(AllFragments==1);
Allpts = [x y];
save data\AllFragments.mat AllFragments;
save data\Allpts.mat Allpts;
save data\L.mat L num;
close(figure(1));
figure('name','Filtered Skeleton (Short Filaments Removed)');
imshow(mat2gray(OriginImg));hold on;
plot(y-R,x-R,'r.');axis off;
% remove very short fragment


% --- Executes on button press in RemoveCrossBtn.
function RemoveCrossBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RemoveCrossBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% remove margin information below

load data\RawSke.mat;
load data\OriginImg.mat;
load data\R.mat;
AllFragments = RawSke;
NOptsMargin = R;
DeleteList = [];

AllFragments(1:NOptsMargin,:) = 0;
AllFragments((end-NOptsMargin):end,:) = 0;
AllFragments(:,1:NOptsMargin) = 0;
AllFragments(:,(end-NOptsMargin):end) = 0;

[x y] = find(AllFragments==1);
Allpts = [x y];
% remove margin information above

% remove crossing points, a N-by-N region located at the base point will be removed
R_Junc = floor((str2num(get(handles.SizofJuncEdit,'String'))-1)/2);
Size_Junc = str2num(get(handles.SizofJuncEdit,'String'));
N1 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)-1,Allpts(:,2)-1));
N2 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)-1,Allpts(:,2)));
N3 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)-1,Allpts(:,2)+1));
N4 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1),Allpts(:,2)-1));
N5 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1),Allpts(:,2)+1));
N6 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)+1,Allpts(:,2)-1));
N7 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)+1,Allpts(:,2)));
N8 = AllFragments(sub2ind(size(AllFragments),Allpts(:,1)+1,Allpts(:,2)+1));

N = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8;

CrPts = Allpts(find(N>=3),1:2);

h = waitbar(0,'Removing crossing points');
for i = 1:size(CrPts,1)
    waitbar(i/size(CrPts,1),h);
    AllFragments(CrPts(i,1)-R_Junc:CrPts(i,1)+R_Junc,CrPts(i,2)-R_Junc:CrPts(i,2)+R_Junc) = 0;
end

close(h);
[x y] = find(AllFragments==1);
Allpts = [x y];

N1 = AllFragments(sub2ind(size(AllFragments),x-1,y-1));
N2 = AllFragments(sub2ind(size(AllFragments),x-1,y));
N3 = AllFragments(sub2ind(size(AllFragments),x-1,y+1));
N4 = AllFragments(sub2ind(size(AllFragments),x,y-1));
N5 = AllFragments(sub2ind(size(AllFragments),x,y+1));
N6 = AllFragments(sub2ind(size(AllFragments),x+1,y-1));
N7 = AllFragments(sub2ind(size(AllFragments),x+1,y));
N8 = AllFragments(sub2ind(size(AllFragments),x+1,y+1));

N = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8;

SinglePts = Allpts(find(N==0),1:2);
% remove single points
h = waitbar(0,'Removing single points');
for i = 1:size(SinglePts,1)
    waitbar(i/size(SinglePts,1),h);
    AllFragments(SinglePts(i,1),SinglePts(i,2)) = 0;
end
close(h);
[x y] = find(AllFragments==1);
Allpts = [x y];
[L num] = bwlabel(AllFragments,8);

RawCrPts = CrPts;
save data\L.mat L num;
save data\AllFragments.mat AllFragments;
save data\Allpts.mat Allpts;
save data\Size_Junc.mat Size_Junc;
save data\RawCrPts.mat RawCrPts;
close(figure(1));
figure('name','Individual Filamentous Fragments');
imshow(mat2gray(OriginImg));hold on;plot(y-R,x-R,'r.');axis off;
% remove clusters of single points


% --- Executes on button press in BackBtn.
function BackBtn_Callback(hObject, eventdata, handles)
% hObject    handle to BackBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close all;
LFT_OFT;



function SizofJuncEdit_Callback(hObject, eventdata, handles)
% hObject    handle to SizofJuncEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of SizofJuncEdit as text
%        str2double(get(hObject,'String')) returns contents of SizofJuncEdit as a double


% --- Executes during object creation, after setting all properties.
function SizofJuncEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SizofJuncEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in AutoThresh.
function AutoThresh_Callback(hObject, eventdata, handles)
% hObject    handle to AutoThresh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
warning off;
load data\OFT_Img;
load data\OriginImg;
load data\R;

DefaultFactor = 1.42;
I = mat2gray(OFT_Img);
t = DefaultFactor*graythresh(I);
if t>=1
    t = graythresh(I);
end
BW = im2bw(I,t);
RawSke = bwmorph(BW,'thin',Inf);
[x y] = find(RawSke==1);
set(handles.ThreshOpt1Edit,'string',t);
msgbox(['The Otsus Threshold Is ',num2str(graythresh(I))]);

close(figure(1));
figure('name','Check Segmented Image (Left) and Extracted Skeleton (Right)');
subplot(1,2,1);
imshow(BW);axis off; title('Segmented Image');
subplot(1,2,2);
imshow(mat2gray(OriginImg));hold on;plot(y-R,x-R,'r.');axis off; title('Original Image with Extracted Skeleton');
scrsz = get(0,'ScreenSize');
set(figure(1),'Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)])
Allpts = [x y];
AllFragments = RawSke;
save data\Allpts.mat Allpts;
save data\RawSke.mat RawSke;
save data\AllFragments.mat AllFragments;


function ThreshOpt1Edit_Callback(hObject, eventdata, handles)
% hObject    handle to ThreshOpt1Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ThreshOpt1Edit as text
%        str2double(get(hObject,'String')) returns contents of ThreshOpt1Edit as a double


% --- Executes during object creation, after setting all properties.
function ThreshOpt1Edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ThreshOpt1Edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ThreshOpt1Btn.
function ThreshOpt1Btn_Callback(hObject, eventdata, handles)
% hObject    handle to ThreshOpt1Btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
t = str2num(get(handles.ThreshOpt1Edit,'String'));
load data\OFT_Img;
load data\OriginImg;
load data\R;

I = mat2gray(OFT_Img);
BW = im2bw(I,t);
RawSke = bwmorph(BW,'thin',Inf);
[x y] = find(RawSke==1);

close(figure(1));
figure('name','Check Segmented Image (Left) and Extracted Skeleton (Right)');
subplot(1,2,1);
imshow(BW);axis off; title('Segmented Image');
subplot(1,2,2);
imshow(mat2gray(OriginImg));hold on;plot(y-R,x-R,'r.');axis off; title('Original Image with Extracted Skeleton'); % the offset of R/4 is determined by trial and error
scrsz = get(0,'ScreenSize');
set(figure(1),'Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)])
Allpts = [x y];
Allpts = [x y];
AllFragments = RawSke;
save data\Allpts.mat Allpts;
save data\RawSke.mat RawSke;
save data\AllFragments.mat AllFragments;


% --- Executes on selection change in MultiCoreList.
function MultiCoreList_Callback(hObject, eventdata, handles)
% hObject    handle to MultiCoreList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns MultiCoreList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MultiCoreList


% --- Executes during object creation, after setting all properties.
function MultiCoreList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MultiCoreList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in IterList.
function IterList_Callback(hObject, eventdata, handles)
% hObject    handle to IterList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns IterList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from IterList


% --- Executes during object creation, after setting all properties.
function IterList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to IterList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in IterationBtn.
function IterationBtn_Callback(hObject, eventdata, handles)
% hObject    handle to IterationBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load data\OriginImg.mat;
load data\AllFragments.mat;
load data\OFT_Img.mat;
load data\R.mat;
load data\ROI_Mask.mat;
load data\NofOrientations_FT.mat;

Thresh = str2num(get(handles.ThreshOpt1Edit,'String'));
JuncSize = str2num(get(handles.SizofJuncEdit,'String'));
MIN_FragmentLength = str2num(get(handles.MinNofPixels,'String'));
iteration = get(handles.IterList,'Value');
Iter_RemoveR = 3; % unit: pixels
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
[L num] = bwlabel(AllFragments);
[x y] = find(AllFragments==1);
Allpts = [x y];
close(figure(1));
figure('name','Ultimate Fragments After Iterative Processing');
imshow(mat2gray(OriginImg));hold on;
plot(y-R,x-R,'r.');axis off;
msgbox('Iterative Extraction of Fragments Done !');

save data\AllFragments.mat AllFragments;
save data\Allpts.mat Allpts;
save data\L.mat L num;


% --- Executes during object deletion, before destroying properties.
function SizofJuncEdit_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to SizofJuncEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
