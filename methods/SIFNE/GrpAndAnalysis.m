% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
%
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function varargout = GrpAndAnalysis(varargin)
% GRPANDANALYSIS MATLAB code for GrpAndAnalysis.fig
%      GRPANDANALYSIS, by itself, creates a new GRPANDANALYSIS or raises the existing
%      singleton*.
%
%      H = GRPANDANALYSIS returns the handle to a new GRPANDANALYSIS or the handle to
%      the existing singleton*.
%
%      GRPANDANALYSIS('CALLBACK',hObject,eventData,handles,...) calls the
%      local
%      function named CALLBACK in GRPANDANALYSIS.M with the given input arguments.
%
%      GRPANDANALYSIS('Property','Value',...) creates a new GRPANDANALYSIS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GrpAndAnalysis_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GrpAndAnalysis_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GrpAndAnalysis

% Last Modified by GUIDE v2.5 06-Jun-2016 16:12:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @GrpAndAnalysis_OpeningFcn, ...
    'gui_OutputFcn',  @GrpAndAnalysis_OutputFcn, ...
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


% --- Executes just before GrpAndAnalysis is made visible.
function GrpAndAnalysis_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GrpAndAnalysis (see VARARGIN)

% Choose default command line output for GrpAndAnalysis
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GrpAndAnalysis wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GrpAndAnalysis_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in PreviewBtn.
function PreviewBtn_Callback(hObject, eventdata, handles)
% hObject    handle to PreviewBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
warning off;
close(figure(1));
load data\AllFragments;
load data\all_tips;
figure('name','Image of All Filamentous Fragments');
imshow(mat2gray(AllFragments));axis off;title('Please double-click the tip where you want to check the region for searching');
[Y1 X1]= ginput(1);
k = dsearchn(all_tips(:,1:2),[X1 Y1]);
k = k(1);
BasePt = all_tips(k,:);
FanR = str2num(get(handles.FanR,'String'));
FanAngle = str2num(get(handles.EditFanAngle,'String'));
FanEdge = FanPreview(FanAngle, AllFragments, FanR, BasePt(4), BasePt(1:2));
close(figure(1));
figure('name','Check the Region for Searching Partners');
imshow(mat2gray(AllFragments));hold on; axis off;
plot(FanEdge(:,2), FanEdge(:,1), 'g', 'LineWidth', 2);


function FanR_Callback(hObject, eventdata, handles)
% hObject    handle to FanR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FanR as text
%        str2double(get(hObject,'String')) returns contents of FanR as a double


% --- Executes during object creation, after setting all properties.
function FanR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FanR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function EditFanAngle_Callback(hObject, eventdata, handles)
% hObject    handle to EditFanAngle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of EditFanAngle as text
%        str2double(get(hObject,'String')) returns contents of EditFanAngle as a double


% --- Executes during object creation, after setting all properties.
function EditFanAngle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to EditFanAngle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function C1_Callback(hObject, eventdata, handles)
% hObject    handle to C1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of C1 as text
%        str2double(get(hObject,'String')) returns contents of C1 as a double


% --- Executes during object creation, after setting all properties.
function C1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to C1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function C3_Callback(hObject, eventdata, handles)
% hObject    handle to C3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of C3 as text
%        str2double(get(hObject,'String')) returns contents of C3 as a double


% --- Executes during object creation, after setting all properties.
function C3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to C3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in QuickSearch.
function QuickSearch_Callback(hObject, eventdata, handles)
% hObject    handle to QuickSearch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(figure(1));
load data\all_tips.mat;
load data\AllFragments;
load data\L.mat;

FanR = str2num(get(handles.FanR,'String'));
FanAngle = str2num(get(handles.EditFanAngle,'String'));

MIN_Angle_Diff = str2num(get(handles.C1,'String'));
MIN_Dist = str2num(get(handles.FanR,'String'));
set(handles.ShortFilamentEdit,'string',MIN_Dist);
MIN_GapAngle_Diff = str2num(get(handles.C3,'String'));

MIN_Info = [MIN_Angle_Diff,MIN_Dist,MIN_GapAngle_Diff];

label_list = all_tips(:,3); % will be updated; help to find another tip of non-first filament during searching
L_GlobalIndex = zeros(size(L));
all_tips(:,7) = (1:size(all_tips,1))';
L_GlobalIndex(sub2ind(size(L_GlobalIndex),all_tips(:,1),all_tips(:,2))) = all_tips(:,7);
% go through all tips and register all searched information

MaxCur = str2num(get(handles.MaxCurEdit,'String'));

new_partner_list = zeros(size(all_tips,1),100); % reserve the memory; a maximum of 100 partner tips allowed
C1weight = str2num(get(handles.C1weightEdit,'String'));
C3weight = str2num(get(handles.C3weightEdit,'String'));

MultiCore = get(handles.MultiCoreList,'Value');
tic;
if MultiCore==1
    h = waitbar(0,'Tip Searching in Progress ...');
    for i = 1:size(all_tips,1)
        waitbar(i/size(all_tips,1),h);
        new_partner_list(i,:) = local_search(L_GlobalIndex, ...            % image of skeleton and its tips are labeled with global index
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
        new_partner_list(i,:) = local_search(L_GlobalIndex, ...            % image of skeleton and its tips are labeled with global index
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

for i = 1:size(new_partner_list,2)
    if sum(new_partner_list(:,i))==0
        new_partner_list(:,i:end) = [];
        break;
    end
end
save data\new_partner_list.mat new_partner_list;

all_tips(:,7) = 1:size(all_tips,1); % add global index to all tips
all_tips(:,8) = ones(size(all_tips,1),1); % this reserves to indicates number of lives

Overlap = get(handles.OverlapList,'Value');

h = waitbar(0,'Add Number of Lives for Each Filament ...');
% add the number of lives & partner tips to all tips
all_tips(:,9:end) = []; % clear previous record
for i = 1:size(all_tips,1)
    waitbar(i/size(all_tips,1),h);
    temp_partner = new_partner_list(i,:);
    temp_partner(find(temp_partner==0)) = [];
    if ~isempty(temp_partner)
        all_tips(i,8) = length(temp_partner);
        all_tips(i,9:(8 + length(temp_partner))) = temp_partner;
    end
end
close(h);
% add the number of lives & partner tips to all tips
[L num] = bwlabel(AllFragments,8);
L_GlobalIndex = zeros(size(L));
L_GlobalIndex(sub2ind(size(L),all_tips(:,1),all_tips(:,2))) = 1:length(all_tips(:,1));
save data\L_GlobalIndex.mat L_GlobalIndex;
msgbox('Tip Search Done ! Please Proceed to GROUPING !');

% assign number of lives to each fragments according to the max number of lives of its two tips
for i = 1:num
    pair_index = find(all_tips(:,3)==i);
    if isempty(pair_index)
        continue;
    end
    all_tips(pair_index(1),8) = max(all_tips(pair_index,8));
    all_tips(pair_index(2),8) = max(all_tips(pair_index,8));
end

all_tips = biDirPairing(all_tips);

if Overlap == 1
    all_tips(:,8) = ones(size(all_tips,1),1); % if overlap is not allowed, the number of lives should be '1'
end

save data\all_tips.mat all_tips;

% ****** structure of all_tips so far ******   #: Number
%           colume1            colume2           colume3         colume4         colume5            colume6        colume7          colume8          colume9               colume10                colume11          ...
% tip1     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip2     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip3     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip4     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% ......
% ****** structure of all_tips so far ******   #: Number
% assign number of lives to each fragments according to the max number of lives of its two tips


% --- Executes on button press in SortingBtn.
function SortingBtn_Callback(hObject, eventdata, handles)
% hObject    handle to SortingBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(figure(1));
load data\all_filament.mat;
load data\all_connects.mat;
load data\L;

maskI = zeros(size(L));
Fullength = 5*size(all_filament,1);
all_sorted_filament = zeros(Fullength,2,size(all_filament,3));
MultiCore = get(handles.MultiCoreList,'Value');
tic;
if MultiCore==1
    h = waitbar(0,'Analysis in Progress ...');
    for i = 1:size(all_filament,3)
        waitbar(i/size(all_filament,3),h);
        [all_sorted_filament(:,:,i)  AnalysisInfo(i,:)]= SortFilament(all_filament(:,:,i), maskI, all_connects(:,:,i), Fullength);
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
        [all_sorted_filament(:,:,i)  AnalysisInfo(i,:)]= SortFilament(all_filament(:,:,i), maskI, all_connects(:,:,i), Fullength);
    end
    delete(gcp);
end
toc;
for i = 1:size(all_sorted_filament,1)
    temp = all_sorted_filament(i,:,:);
    if(sum(temp(:)))==0
        all_sorted_filament(i:end,:,:) = [];
        break;
    end
end
msgbox('Analysis and Sorting Done !');

save data\all_sorted_filament.mat all_sorted_filament;
save data\AnalysisInfo.mat AnalysisInfo;


function EditPixelSize_Callback(hObject, eventdata, handles)
% hObject    handle to EditPixelSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of EditPixelSize as text
%        str2double(get(hObject,'String')) returns contents of EditPixelSize as a double


% --- Executes during object creation, after setting all properties.
function EditPixelSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to EditPixelSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in BackBtn.
function BackBtn_Callback(hObject, eventdata, handles)
% hObject    handle to BackBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close all;
SegmentB4Grouping;


function ShortFilamentEdit_Callback(hObject, eventdata, handles)
% hObject    handle to ShortFilamentEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ShortFilamentEdit as text
%        str2double(get(hObject,'String')) returns contents of ShortFilamentEdit as a double


% --- Executes during object creation, after setting all properties.
function ShortFilamentEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ShortFilamentEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in RemoveShortBtn.
function RemoveShortBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RemoveShortBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(figure(1));
close(figure(2));

load data\AnalysisInfo;
load data\all_sorted_filament;
load data\AllFragments.mat;
load data\all_connects.mat;

ShortFilament= str2num(get(handles.ShortFilamentEdit,'String'));

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
RemoveUngrp = get(handles.RemoveUngrp,'Value');
if RemoveUngrp==1
    AllFragments = im2bw(AllFragments);
    RemoveIdx = [];
    h = waitbar(0,'Removing Ungrouped Fragments...');
    for i = 1:size(all_sorted_filament,3)
        waitbar(i/size(all_sorted_filament,3),h);
        temp = all_sorted_filament(:,:,i);
        temp(find(temp(:,1)==0),:) = [];
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
save data\AnalysisInfo.mat AnalysisInfo;
save data\all_sorted_filament.mat all_sorted_filament;
save data\all_connects_shortlist.mat all_connects_shortlist;

% --- Executes on button press in QuickGrp.
function QuickGrp_Callback(hObject, eventdata, handles)
% hObject    handle to QuickGrp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% filamentous fragment grouping starts
close(figure(1));
load data\L.mat;
load data\all_tips.mat;

NofLives = all_tips(:,8);
NofLives(find(NofLives==0)) = 1;
all_tips(:,8) = NofLives;
% assign number of lives to each fragments according to the max number of lives of its two tips

new_partner_list = all_tips(:,9:end); % update the list of partners

all_filament = [];
all_connects = [];
[x y] = find(L~=0);
Lpts = [x, y, L(sub2ind(size(L),x,y))];
h = waitbar(0,'Grouping in Progress');
tic;
for i = 1:num
    waitbar(i/num,h);
    L_list = [];
    label_list = all_tips(:,3); % labels (corresponding to L) of all fragments the global index of which corresponds to all_tips
    xx = Lpts(find(Lpts(:,3)==i),1);
    yy = Lpts(find(Lpts(:,3)==i),2);
    if ~isempty(xx) % if this fragment still exists
        tips_index = find(all_tips(:,3)==i);% find global index
        newSinglefilament = []; % used to store a single filament
        newSinglefilament = [newSinglefilament; [xx yy]]; % add up to the single filament
        L_list = [L_list  i];
        tip_dir1 = tips_index(1); % change to the first tip
        tip_dir2 = tips_index(2);
        connect_tips = all_tips(tip_dir1,1:2); % the design of this connect_tips is for quick filling between gaps later
        if NofLives(tip_dir1)<2
            Lpts(find(Lpts(:,3)==i),3)=0;
            label_list(tips_index(1)) = 0;
            label_list(tips_index(2)) = 0;
            new_partner_list(find(new_partner_list==tip_dir1)) = 0;
            new_partner_list(find(new_partner_list==tip_dir2)) = 0;
        else
            NofLives(tips_index(1)) = NofLives(tips_index(1)) - 1; % reduce its number of lives by 1 if it is used
            NofLives(tips_index(2)) = NofLives(tips_index(2)) - 1; % reduce its number of lives by 1 if it is used
        end
        while 1+1==2
            can_index = new_partner_list(tip_dir1,:); % get global indices of possible partner tips
            can_index(find(can_index==0)) = []; % remove zeros
            if isempty(can_index) || ~isempty(find(L_list==all_tips(can_index(1),3)))  % if the current tip has no possible partner tip
                break;
            else
                optimal_index = can_index(1);
                xx = Lpts(find(Lpts(:,3)==label_list(optimal_index)),1);
                yy = Lpts(find(Lpts(:,3)==label_list(optimal_index)),2);
                newSinglefilament = [newSinglefilament; [xx yy]]; % add new fragment to the current single filament
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
msgbox('Grouping Done !');
close(h);
close(figure(1));
% filamentous fragment grouping ends
save data\all_filament.mat all_filament;
save data\all_connects.mat all_connects;
all_connects_shortlist = all_connects;
save data\all_connects_shortlist.mat all_connects_shortlist;


% --- Executes on selection change in AnalysisList.
function AnalysisList_Callback(hObject, eventdata, handles)
% hObject    handle to AnalysisList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns AnalysisList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from AnalysisList


% --- Executes during object creation, after setting all properties.
function AnalysisList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AnalysisList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in RunAnalysisBtn.
function RunAnalysisBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RunAnalysisBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

A = get(handles.AnalysisList,'Value');
switch A
    case 1
        warning off;
        close(figure(1));close(figure(2));
        load data\R.mat;
        load data\AllFragments.mat;
        load data\L.mat;
        load data\all_sorted_filament.mat all_sorted_filament;
        load data\AnalysisInfo.mat;
        load data\ROI_Mask;
        % reconstruct network
        allpts_IncludeDup = [];
        Overlay_Map = zeros(size(L));
        h = waitbar(0,'Retrieve All Filaments');
        for i = 1:size(all_sorted_filament,3)
            waitbar(i/size(all_sorted_filament,3),h);
            temp1 = all_sorted_filament(:,:,i); % get all coordinates
            temp1(find(temp1(:,1)==0),:) = []; % remove zeros
            allpts_IncludeDup = [allpts_IncludeDup;temp1];
            % check points used more than once
            Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) = Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) + 1;
        end
        close(h);
        
        figure(1);
        imshow(mat2gray(AllFragments((R+1):(size(AllFragments,1)-R), (R+1):(size(AllFragments,2)-R))));hold on;axis off;
        screen_size = get(0, 'ScreenSize'); set(figure(1), 'Position', [0 0 screen_size(3) screen_size(4) ] );
        ColorList = rand(32,3);
        temp = 1:size(all_sorted_filament,3);
        for i = 1:32
            idx = find(mod(temp,32)==(i-1));
            temp1 = all_sorted_filament(:,1,idx);  temp1 = temp1(:);  temp1(find(temp1==0)) = [];
            temp2 = all_sorted_filament(:,2,idx);  temp2 = temp2(:);  temp2(find(temp2==0)) = [];
            plot(temp2-R,temp1-R,'.','color',ColorList(i,:),'MarkerSize',6);hold on;
        end
        Overlap = get(handles.OverlapList,'Value');
        if Overlap==2
            [x y] = find(Overlay_Map>1);
            plot(y-R,x-R,'r.','MarkerSize',6);hold on;
        end
        B = bwboundaries(ROI_Mask);
        B = B{1};
        plot(B(:,2),B(:,1),'.','color',[1 1 1],'MarkerSize',6);hold on;
        
        mkdir result;
        saveas(figure(1),'result\Extracted_Filaments.fig');
    case 2
        warning off;
        close(figure(1));close(figure(2));
        load data\R.mat;
        load data\OriginImg.mat;
        load data\L.mat;
        load data\all_sorted_filament.mat all_sorted_filament;
        load data\ROI_Mask;
        PixelSize = str2num(get(handles.EditPixelSize,'String'));
        
        
        Overlay_Map = zeros(size(L));
        h = waitbar(0,'Checking Junctions ...');
        for i = 1:size(all_sorted_filament,3)
            waitbar(i/size(all_sorted_filament,3),h);
            temp1 = all_sorted_filament(:,:,i);
            temp1(find(temp1(:,1)==0),:) = [];
            Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) = Overlay_Map(sub2ind(size(L),temp1(:,1),temp1(:,2))) + 1;
        end
        close(h);
        [x y] = find(Overlay_Map>1);
        
        CroMap = zeros(size(L));
        CroMap(sub2ind(size(L),x,y)) = 1;
        for i = 1:length(x)
            temp = CroMap((x(i)-1):(x(i)+1), (y(i)-1):(y(i)+1));
            if sum(temp(:))>1
                CroMap(x(i),y(i)) = 0;
            end
        end
        [x y] = find(CroMap==1);
        NewCrPts = [x y];
        save data\NewCrPts.mat NewCrPts;
        
        disF=bwdist(bwmorph(ROI_Mask,'remove'));
        mask = ROI_Mask;
        mask = single(mask);
        mask(mask==0)=0;
        disF = disF.*mask;
        disF = disF*PixelSize;
        
        figure(1);
        imagesc(disF);colormap(jet);hold on;axis off;axis image;colorbar;
        [xx yy] = find(Overlay_Map~=0);
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
        figure(2);
        %         histogram(d);%axis square;
        histfit(d,50,'kernel');xlim([0 inf]);
        ylabel('Frequency');
        xlabel('Distance to Cell Edge (\mum)');
        title('Distribution of Junctions');
        mkdir result;
        saveas(figure(1),'result\Distribution_Junctions.fig');
        saveas(figure(2),'result\Distribution_Junctions_Analysis.fig');
        
        
    case 3
        warning off;
        close(figure(1));close(figure(2));
        
        load data\all_sorted_filament.mat;
        load data\AnalysisInfo.mat;
        load data\L;
        load data\ROI_Mask;
        % plot of information of all filaments
        PixelSize = str2num(get(handles.EditPixelSize,'String'));
        figure(1);
        h=rose(pi*AnalysisInfo(:,1)/180,30);axis off;
        set(h,'linewidth',3)
        axis square;
        title('Histogram of Filaments Orientations');
        
        mkdir result;
        saveas(figure(1),'result\Histogram_of_Orientations.fig');
        
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
        xlabel('Distance to Cell Edge (\mum)');ylabel('-90 degrees to 90 degrees');
        title('Distribution of Filament Orientations');
        saveas(figure(2),'result\Distribution_Orientations.fig');
        
        
    case 4
        warning off;
        close(figure(1));close(figure(2));
        load data\all_sorted_filament;
        load data\ROI_Mask;
        load data\AllFragments;
        curR = round(str2num(get(handles.FanR,'String'))/2);
        PixelSize = str2num(get(handles.EditPixelSize,'String'));
        
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
        figure(1);imagesc(M); colormap(jet);colorbar; title('Distribution of Curvatures');axis off;axis image;
          
        figure(2);
        subplot(1,2,1);
        histfit(all_mean_cur,50,'kernel');xlim([0 inf]);
        xlabel('Curvature (unit: \mum^-^1)');ylabel('Frequency');
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
        save data\all_filament_curs.mat all_filament_curs;
        mkdir result;
        saveas(figure(1),'result\Distribution_of_Curvatures.fig');
        saveas(figure(2),'result\Distribution_of_Curvatures_Analysis.fig');
        
    case 5
        warning off;
        close(figure(1));close(figure(2));
        load data\AnalysisInfo;
        load data\all_sorted_filament;
        load data\NewCrPts;
        load data\Size_Junc;
        load data\ROI_Mask;
        load data\L.mat;
        load data\all_tips.mat;
        load data\all_connects.mat;
        load data\all_connects_shortlist.mat;
        
        % generate information of all filaments
        PixelSize = str2num(get(handles.EditPixelSize,'String'));
        titles = {'Filament ID','1st X pos','1st Y pos','last X pos','last Y pos','Orientation','Total Length','End-to-End Distance','Centroid X','Centroid Y'};
        titles = [titles,repmat({'X','Y'},[1 size(all_sorted_filament,1)]) ];
        InfoExcel = zeros(size(all_sorted_filament,3)*2, size(all_sorted_filament,1)*2+9);
        
        h = waitbar(0,'Generating information of composite filaments ...');
        for i = 1:size(all_sorted_filament,3)
            waitbar(i/size(all_sorted_filament,3),h);
            curr_filament = all_sorted_filament(:,:,i);
            curr_filament(find(curr_filament(:,1)==0),:) = [];
            InfoExcel((i*2-1), 1:9) = [curr_filament(1,:), curr_filament(end,:), AnalysisInfo(i,:)];
            temp = zeros(1,size(all_sorted_filament,1)*2);
            temp((1:size(all_sorted_filament,1))*2-1) = all_sorted_filament(:,1,i);
            temp((1:size(all_sorted_filament,1))*2) = all_sorted_filament(:,2,i);
            InfoExcel((i*2-1), 10:end) = temp;
            ks = dsearchn(curr_filament,NewCrPts);
            for j = 1:length(ks)
                if pdist([curr_filament(ks(j),:); NewCrPts(j,:)])<=Size_Junc
                    InfoExcel((i*2), ((9+ks(j)*2-1):(9+ks(j)*2))) = NewCrPts(j,:);
                end
            end
        end
        FragID = 1:size(all_sorted_filament,3);
        FragID = [FragID; FragID];
        FragID = FragID(:);
        InfoExcel = [FragID, InfoExcel];
        save data\InfoExcel.mat InfoExcel;
        close(h);
        InfoExcel = [titles;num2cell(InfoExcel)];
        
        
        % ------ generate fragment information ------
        FragmentInfo = zeros(num,size(L,1));
        MultiCore = get(handles.MultiCoreList,'Value');
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
            load data\L.mat;
            L = L;
            load data\all_tips.mat;
            all_tips = all_tips;
            parfor i = 1:num
                FragmentInfo(i,:) = GenFragmentInfo(L,i,all_tips,FragmentInfo(i,:));
            end
        end
        
        
        sumlist = sum(FragmentInfo,1);
        FragmentInfo(:,(find(sumlist==0))) = [];
        save data\FragmentInfo.mat FragmentInfo;
        MaxFragL = size(FragmentInfo,2)-6;
        display('Fragment Information Generated !');
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
            load data\L.mat;
            L = L;
            parfor i = 1:size(all_connects,3)
                LinkInfo(:,:,i) = GenLinkageInfo(i,L,all_connects(:,:,i),LinkInfo(:,:,i));
            end
        end
        
        display('Linkage1 Information Generated !');
        
        
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
        display('Linkage1 Integration Done !');
        sumlist = sum(LinkageInfo1,1);
        LinkageInfo1(find(LinkageInfo1(:,1)==0),:) = [];
        LinkageInfo1(:,(find(sumlist==0))) = [];
        save data\LinkageInfo1.mat LinkageInfo1;
        titles = {'Composite Filament ID','Fragment ID'};
        titles = [titles,repmat({'X','Y'},[1 (size(LinkageInfo1,2)-2)/2]) ];
        LinkageInfo1 = [titles;num2cell(LinkageInfo1)];
        
        
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
        
        display('Linkage2 Information Generated !');
        
        
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
        display('Linkage2 Integration Done !');
        sumlist = sum(LinkageInfo2,1);
        LinkageInfo2(find(LinkageInfo2(:,1)==0),:) = [];
        LinkageInfo2(:,(find(sumlist==0))) = [];
        save data\LinkageInfo2.mat LinkageInfo2;
        titles = {'Composite Filament ID','Fragment ID'};
        titles = [titles,repmat({'X','Y'},[1 (size(LinkageInfo2,2)-2)/2]) ];
        LinkageInfo2 = [titles;num2cell(LinkageInfo2)];
        
        mkdir result;
        % write inforamtion to excel
        cd result;
        
        xlswrite('IntegratedInfo.xlsx',InfoExcel,1,'A1');
        xlswrite('IntegratedInfo.xlsx',FragmentInfo,2,'A1');
        xlswrite('IntegratedInfo.xlsx',LinkageInfo1,3,'A1');
        xlswrite('IntegratedInfo.xlsx',LinkageInfo2,4,'A1');
        curDir = pwd;
        filaname = [curDir,'\IntegratedInfo.xlsx'];
        e = actxserver('Excel.Application');
        ewb = e.Workbooks.Open(filaname);
        ewb.Worksheets.Item(1).Name = 'Ultimate Filaments';
        ewb.Worksheets.Item(2).Name = 'Fragment Info';
        ewb.Worksheets.Item(3).Name = 'Linkage Info1';
        ewb.Worksheets.Item(4).Name = 'Linkage Info2';
        ewb.Save
        ewb.Close(false);
        e.Quit;
        cd ..;
        msgbox('Excel Generated !');
        
end


% --- Executes on button press in CompleteSaveBtn.
function CompleteSaveBtn_Callback(hObject, eventdata, handles)
% hObject    handle to CompleteSaveBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FanR = str2num(get(handles.FanR,'String'));
EditFanAngle = str2num(get(handles.EditFanAngle,'String'));
C1 = str2num(get(handles.C1,'String'));
C1weight = str2num(get(handles.C1weightEdit,'String'));
C2 = str2num(get(handles.FanR,'String'));
C3weight = str2num(get(handles.C3weightEdit,'String'));
C3 = str2num(get(handles.C3,'String'));
EditPixelSize = str2num(get(handles.EditPixelSize,'String'));
ShortFilamentEdit = str2num(get(handles.ShortFilamentEdit,'String'));
MaxCur = str2num(get(handles.MaxCurEdit,'String'));

% save user settings
fileID = fopen('UserSettings\GroupingSettings.txt','w');
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



function MaxCurEdit_Callback(hObject, eventdata, handles)
% hObject    handle to MaxCurEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of MaxCurEdit as text
%        str2double(get(hObject,'String')) returns contents of MaxCurEdit as a double


% --- Executes during object creation, after setting all properties.
function MaxCurEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MaxCurEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in AutoSetCondBtn.
function AutoSetCondBtn_Callback(hObject, eventdata, handles)
% hObject    handle to AutoSetCondBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
pixelsize = str2num(get(handles.EditPixelSize,'String'));
MaxCur = str2num(get(handles.MaxCurEdit,'String'));
FanRadius = 1/MaxCur/pixelsize;
FanAngle= 360/2/pi;
set(handles.EditFanAngle,'string',FanAngle);
set(handles.C1,'string',FanAngle);
set(handles.C3,'string',FanAngle/2);
set(handles.FanR,'string',FanRadius);
set(handles.ShortFilamentEdit,'string',FanRadius);


function C3weightEdit_Callback(hObject, eventdata, handles)
% hObject    handle to C3weightEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of C3weightEdit as text
%        str2double(get(hObject,'String')) returns contents of C3weightEdit as a double


% --- Executes during object creation, after setting all properties.
function C3weightEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to C3weightEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function C1weightEdit_Callback(hObject, eventdata, handles)
% hObject    handle to C1weightEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of C1weightEdit as text
%        str2double(get(hObject,'String')) returns contents of C1weightEdit as a double


% --- Executes during object creation, after setting all properties.
function C1weightEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to C1weightEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


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


% --- Executes on selection change in OverlapList.
function OverlapList_Callback(hObject, eventdata, handles)
% hObject    handle to OverlapList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns OverlapList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from OverlapList


% --- Executes during object creation, after setting all properties.
function OverlapList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to OverlapList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in RemoveUngrp.
function RemoveUngrp_Callback(hObject, eventdata, handles)
% hObject    handle to RemoveUngrp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of RemoveUngrp


% --- Executes on button press in OptManualGUIBtn.
function OptManualGUIBtn_Callback(hObject, eventdata, handles)
% hObject    handle to OptManualGUIBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FanR = str2num(get(handles.FanR,'String'));
save data\FanR.mat FanR;
ManCorr;
