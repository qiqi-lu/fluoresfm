% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function varargout = ManCorr(varargin)
% MANCORR MATLAB code for ManCorr.fig
%      MANCORR, by itself, creates a new MANCORR or raises the existing
%      singleton*.
%
%      H = MANCORR returns the handle to a new MANCORR or the handle to
%      the existing singleton*.
%
%      MANCORR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MANCORR.M with the given input arguments.
%
%      MANCORR('Property','Value',...) creates a new MANCORR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ManCorr_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ManCorr_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ManCorr

% Last Modified by GUIDE v2.5 22-Apr-2016 16:27:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @ManCorr_OpeningFcn, ...
    'gui_OutputFcn',  @ManCorr_OutputFcn, ...
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


% --- Executes just before ManCorr is made visible.
function ManCorr_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ManCorr (see VARARGIN)

% Choose default command line output for ManCorr
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;


handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ManCorr wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ManCorr_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in ManCorrConnectBtn.
function ManCorrConnectBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ManCorrConnectBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
figure(1);
[y x] = ginput(1);
Dlist = sqrt((M_all_tips(:,1)-x).^2 + (M_all_tips(:,2)-y).^2);
Dlist = abs(Dlist);
idx = find(Dlist==min(Dlist));idx = idx(1);
tip1 = M_all_tips(idx,:);
plot(tip1(2),tip1(1),'r+','MarkerSize',30);

[y x] = ginput(1);
Dlist = sqrt((M_all_tips(:,1)-x).^2 + (M_all_tips(:,2)-y).^2);
Dlist = abs(Dlist);
idx = find(Dlist==min(Dlist));idx = idx(1);
tip2 = M_all_tips(idx,:);
plot(tip2(2),tip2(1),'r+','MarkerSize',30);

if tip1(3)==tip2(3)
    msgbox('You chose the head and tail of the same filament !');
else
    F1 = M_all_pts(find(M_all_pts(:,3)==tip1(3)),1:2);
    F2 = M_all_pts(find(M_all_pts(:,3)==tip2(3)),1:2);
    temp = zeros(size(M_OriginMargin));
    temp(sub2ind(size(temp),[F1(:,1);F2(:,1)],[F1(:,2);F2(:,2)])) = 1;
    temp = filline(temp,1,tip1(1),tip1(2),tip2(1),tip2(2));
    
    [x y] = find(temp==1);
    for k = 1:length(x)
        localtemp = temp((x(k)-1):(x(k)+1), (y(k)-1):(y(k)+1));
        if sum(localtemp(:))==2
            break;
        end
    end
    changing = [x(k)  y(k)];
    x(k) = []; y(k) = [];
    Fnew = changing;
    while ~isempty(x)
        k = dsearchn([x y],changing);
        changing = [x(k)  y(k)];
        Fnew = [Fnew; changing];
        x(k) = []; y(k) = [];
    end
    
    M_asf(:,:,[tip1(3), tip2(3)]) = [];
    M_asf(1:size(Fnew,1),1:2,size(M_asf,3)+1) = Fnew;
    
    h = waitbar(0,'updating ...');
    M_all_pts = [];
    M_all_tips = [];
    for j = 1:size(M_asf,3)
        waitbar(j/size(M_asf,3),h);
        F = M_asf(:,:,j);
        F(find(F(:,1)==0),:) = [];
        M_all_pts = [M_all_pts; [F  j*ones(size(F,1),1)]];
        M_all_tips = [M_all_tips; [F(1,:)  j;F(end,:)  j]];
    end
    close(h);
    figure(1);
    plot(tip1(2),tip1(1),'.','color',[0 0 0]+M_OriginMargin(tip1(1),tip1(2)),'MarkerSize',30);
    plot(tip2(2),tip2(1),'.','color',[0 0 0]+M_OriginMargin(tip2(1),tip2(2)),'MarkerSize',30);
    plot(Fnew(:,2),Fnew(:,1),'c.');
end





% --- Executes on button press in ManCorrCompleteBtn.
function ManCorrCompleteBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ManCorrCompleteBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
close(figure(1));
h = waitbar(0,'Re-Analysis in progress ...');
for i = 1:size(M_asf,3)
    waitbar(i/size(M_asf,3),h);
    NewAnalysisInfo(i,:) = ManCorrSortFilament(M_asf(:,:,i));
end
close(h);
msgbox('Manual Correction Done !');
AnalysisInfo = NewAnalysisInfo;
all_sorted_filament = M_asf;
save data\AnalysisInfo.mat AnalysisInfo;
save data\all_sorted_filament.mat all_sorted_filament;




% --- Executes on button press in ManCorrNextBtn.
function ManCorrNextBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ManCorrNextBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
counter = counter+1;

key = get(gcf,'CurrentKey');
if(strcmp (key , 'n'))
    pushbutton1_Callback(hObject, eventdata, handles)
end

warning off;
while isempty(find(M_all_tips(find(M_all_tips(:,1)==Minitial_all_tips(counter,1)),2)==Minitial_all_tips(counter,2)))
    display(['Current Correction: Tip ',num2str(counter),'/',num2str(size(Minitial_all_tips,1)),'. (skipped)']);
    counter = counter+1;
    if counter==size(Minitial_all_tips,1)
        break;
    end
end

if counter~=size(Minitial_all_tips,1)
    close(figure(1));
    figure(1);
    imshow(M_OriginMargin);hold on;axis off;
    plot(M_all_pts(:,2),M_all_pts(:,1),'c.');
    plot(M_all_tips(:,2),M_all_tips(:,1),'r.','MarkerSize',30);
    plot(Minitial_all_tips(counter,2),Minitial_all_tips(counter,1),'ro','MarkerSize',30);
    
    wing = 250;
    upper = Minitial_all_tips(counter,1)-wing;
    lower = Minitial_all_tips(counter,1)+wing;
    left = Minitial_all_tips(counter,2)-wing;
    right = Minitial_all_tips(counter,2)+wing;
    if upper<=0
        upper = 1;
    end
    if left<=0
        left = 1;
    end
    if lower>size(M_OriginMargin,1)
        lower = size(M_OriginMargin,1);
    end
    if right>size(M_OriginMargin,2)
        right = size(M_OriginMargin,2);
    end
    axis([left  right  upper lower]);
    display(['Current Correction: Tip ',num2str(counter),'/',num2str(size(Minitial_all_tips,1)),'.']);
else
    msgbox('You have finished all tips !');
end




% --- Executes on button press in ManCorrHighFilaBtn.
function ManCorrHighFilaBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ManCorrHighFilaBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
figure(1);
[y x] = ginput(1);
Dlist = sqrt((M_all_pts(:,1)-x).^2 + (M_all_pts(:,2)-y).^2);
Dlist = abs(Dlist);
idx = find(Dlist==min(Dlist));idx = idx(1);
clickF = M_all_pts(find(M_all_pts(:,3)==M_all_pts(idx,3)),1:2);
plot(clickF(:,2),clickF(:,1),'r.');




% --- Executes on button press in ManCorrBrkBtn.
function ManCorrBrkBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ManCorrBrkBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
figure(1);
[y x] = ginput(1);
Dlist = sqrt((M_all_pts(:,1)-x).^2 + (M_all_pts(:,2)-y).^2);
Dlist = abs(Dlist);
idx = find(Dlist==min(Dlist));idx = idx(1);
clickF = M_all_pts(find(M_all_pts(:,3)==M_all_pts(idx,3)),1:2);
plot(clickF(:,2),clickF(:,1),'r.');

h = imfreehand;
pos = h.getPosition();bw = zeros(size(M_OriginMargin));
pos = [pos(:,2) pos(:,1)];
bw(sub2ind(size(bw),round(pos(:,1)),round(pos(:,2)))) = 1;
bw = roipoly(bw,pos(:,2),pos(:,1));
pts_remove = []; newtips = [];
if bw(sub2ind(size(bw),clickF(1,1),clickF(1,2)))==1 && bw(sub2ind(size(bw),clickF(end,1),clickF(end,2)))==1
    M_asf(:,:,M_all_pts(idx,3)) = [];
    h = waitbar(0,'updating ...');
    M_all_pts = [];
    M_all_tips = [];
    for j = 1:size(M_asf,3)
        waitbar(j/size(M_asf,3),h);
        F = M_asf(:,:,j);
        F(find(F(:,1)==0),:) = [];
        M_all_pts = [M_all_pts; [F  j*ones(size(F,1),1)]];
        M_all_tips = [M_all_tips; [F(1,:)  j;F(end,:)  j]];
    end
    close(h);
    figure(1);
    pts_remove = clickF;
elseif bw(sub2ind(size(bw),clickF(1,1),clickF(1,2)))==0 && bw(sub2ind(size(bw),clickF(end,1),clickF(end,2)))==0
    templist = bw(sub2ind(size(bw),clickF(:,1),clickF(:,2)));
    templist = templist(:);
    flaglist = [templist(2:end);templist(end)] - templist;
    F1 = clickF(1:find(flaglist==1),1:2);
    F2 = clickF((find(flaglist==-1)+1):end,1:2);

    M_asf(:,:,M_all_pts(idx,3)) = [];
    
    M_asf(1:size(F1,1),1:2,size(M_asf,3)+1) = F1;
    M_asf(1:size(F2,1),1:2,size(M_asf,3)+1) = F2;
    
    h = waitbar(0,'updating ...');
    M_all_pts = [];
    M_all_tips = [];
    for j = 1:size(M_asf,3)
        waitbar(j/size(M_asf,3),h);
        F = M_asf(:,:,j);
        F(find(F(:,1)==0),:) = [];
        M_all_pts = [M_all_pts; [F  j*ones(size(F,1),1)]];
        M_all_tips = [M_all_tips; [F(1,:)  j;F(end,:)  j]];
    end
    close(h);
    figure(1);
    pts_remove = clickF((find(flaglist==1)+1):find(flaglist==-1),:);
    newtips = [F1(1,:);F1(end,:);F2(1,:);F2(end,:)];
else
    templist = bw(sub2ind(size(bw),clickF(:,1),clickF(:,2)));
    templist = templist(:);
    flaglist = [templist(2:end);templist(end)] - templist;
    if ~isempty(find(flaglist==1))
        F1 = clickF((find(flaglist==1)+1):end,:);
        F2 = clickF(1:find(flaglist==1),:);
    else
        F1 = clickF(1:find(flaglist==-1),:);
        F2 = clickF((find(flaglist==-1)+1):end,:);
    end

    M_asf(:,:,M_all_pts(idx,3)) = [];
    M_asf(1:size(F2,1),1:2,size(M_asf,3)+1) = F2;
    h = waitbar(0,'updating ...');
    M_all_pts = [];
    M_all_tips = [];
    for j = 1:size(M_asf,3)
        waitbar(j/size(M_asf,3),h);
        F = M_asf(:,:,j);
        F(find(F(:,1)==0),:) = [];
        M_all_pts = [M_all_pts; [F  j*ones(size(F,1),1)]];
        M_all_tips = [M_all_tips; [F(1,:)  j;F(end,:)  j]];
    end
    close(h);
    figure(1);
    pts_remove = F1;newtips = [F2(1,:);F2(end,:)];
end
plot(clickF(:,2),clickF(:,1),'c.');
for i = 1:size(pts_remove,1)
    plot(pts_remove(i,2),pts_remove(i,1),'.','color',[0 0 0]+M_OriginMargin(pts_remove(i,1),pts_remove(i,2)));
end
if ~isempty(newtips)
    plot(newtips(:,2),newtips(:,1),'r.','MarkerSize',30);
end



% --- Executes on button press in ManCorrNewFilaBtn.
function ManCorrNewFilaBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ManCorrNewFilaBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
figure(1);
h = impoly('closed',0);pos = h.getPosition();
pos = round([pos(:,2), pos(:,1)]);
temp = zeros(size(M_OriginMargin));
for i = 1:(size(pos,1)-1)
    temp = filline(temp,1,pos(i,1),pos(i,2),pos(i+1,1),pos(i+1,2));
end
[x y] = find(temp==1);
for k = 1:length(x)
    localtemp = temp((x(k)-1):(x(k)+1), (y(k)-1):(y(k)+1));
    if sum(localtemp(:))==2
        break;
    end
end
changing = [x(k)  y(k)];
x(k) = []; y(k) = [];
Fnew = changing;
while ~isempty(x)
    k = dsearchn([x y],changing);
    changing = [x(k)  y(k)];
    Fnew = [Fnew; changing];
    x(k) = []; y(k) = [];
end

M_asf(1:size(Fnew,1),1:2,size(M_asf,3)+1) = Fnew;
h = waitbar(0,'updating ...');
M_all_pts = [];
M_all_tips = [];
for j = 1:size(M_asf,3)
    waitbar(j/size(M_asf,3),h);
    F = M_asf(:,:,j);
    F(find(F(:,1)==0),:) = [];
    M_all_pts = [M_all_pts; [F  j*ones(size(F,1),1)]];
    M_all_tips = [M_all_tips; [F(1,:)  j;F(end,:)  j]];
end
close(h);
figure(1);
plot(Fnew(:,2),Fnew(:,1),'c.');
plot(Fnew(1,2),Fnew(1,1),'r.','MarkerSize',30);
plot(Fnew(end,2),Fnew(end,1),'r.','MarkerSize',30);


% --- Executes on button press in InitialBtn.
function InitialBtn_Callback(hObject, eventdata, handles)
% hObject    handle to InitialBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
counter = 1;
close(figure(1));
load data\R.mat;
load data\OriginImg.mat;
load data\all_sorted_filament.mat;
M_OriginMargin = zeros(size(OriginImg,1)+R+R, size(OriginImg,2)+R+R);
M_OriginMargin((R+1):(size(OriginImg,1)+R),(R+1):(size(OriginImg,2)+R)) = mat2gray(OriginImg);
M_asf = all_sorted_filament;
h = waitbar(0,'preparing ...');
M_all_pts = [];
M_all_tips = [];
for j = 1:size(M_asf,3)
    waitbar(j/size(M_asf,3),h);
    F = M_asf(:,:,j);
    F(find(F(:,1)==0),:) = [];
    M_all_pts = [M_all_pts; [F  j*ones(size(F,1),1)]];
    M_all_tips = [M_all_tips; [F(1,:)  j;F(end,:)  j]];
end
Minitial_all_tips = M_all_tips;
close(h);
figure(1);
imshow(M_OriginMargin);hold on;axis off;
plot(M_all_pts(:,2),M_all_pts(:,1),'c.');
plot(M_all_tips(:,2),M_all_tips(:,1),'r.','MarkerSize',30);
plot(Minitial_all_tips(counter,2),Minitial_all_tips(counter,1),'ro','MarkerSize',30);
wing = 250;
upper = Minitial_all_tips(counter,1)-wing;
lower = Minitial_all_tips(counter,1)+wing;
left = Minitial_all_tips(counter,2)-wing;
right = Minitial_all_tips(counter,2)+wing;
if upper<=0
    upper = 1;
end
if left<=0
    left = 1;
end
if lower>size(M_OriginMargin,1)
    lower = size(M_OriginMargin,1);
end
if right>size(M_OriginMargin,2)
    right = size(M_OriginMargin,2);
end
axis([left  right  upper lower]);
display(['Current Correction: Tip ',num2str(counter),'/',num2str(size(Minitial_all_tips,1)),'.']);




% --- Executes on button press in tempcomBtn.
function tempcomBtn_Callback(hObject, eventdata, handles)
% hObject    handle to tempcomBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
mkdir ManCorrTempInfo;
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;

save ManCorrTempInfo\M_asf.mat M_asf;
save ManCorrTempInfo\M_OriginMargin.mat M_OriginMargin;
save ManCorrTempInfo\M_all_pts.mat M_all_pts;
save ManCorrTempInfo\M_all_tips.mat M_all_tips;
save ManCorrTempInfo\Minitial_all_tips.mat Minitial_all_tips;
save ManCorrTempInfo\counter.mat counter;
msgbox('Information Saved ! Take Rest and Continue !');







% --- Executes on button press in tempinitialBtn.
function tempinitialBtn_Callback(hObject, eventdata, handles)
% hObject    handle to tempinitialBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;
load ManCorrTempInfo\M_asf.mat;
load ManCorrTempInfo\M_OriginMargin.mat;
load ManCorrTempInfo\M_all_pts.mat;
load ManCorrTempInfo\M_all_tips.mat;
load ManCorrTempInfo\Minitial_all_tips.mat;
load ManCorrTempInfo\counter.mat;

close(figure(1));
load data\R.mat;
load data\OriginImg.mat;
M_OriginMargin = zeros(size(OriginImg,1)+R+R, size(OriginImg,2)+R+R);
M_OriginMargin((R+1):(size(OriginImg,1)+R),(R+1):(size(OriginImg,2)+R)) = mat2gray(OriginImg);
h = waitbar(0,'preparing ...');
M_all_pts = [];
M_all_tips = [];
for j = 1:size(M_asf,3)
    waitbar(j/size(M_asf,3),h);
    F = M_asf(:,:,j);
    F(find(F(:,1)==0),:) = [];
    M_all_pts = [M_all_pts; [F  j*ones(size(F,1),1)]];
    M_all_tips = [M_all_tips; [F(1,:)  j;F(end,:)  j]];
end
Minitial_all_tips = M_all_tips;
close(h);
figure(1);
imshow(M_OriginMargin);hold on;axis off;
plot(M_all_pts(:,2),M_all_pts(:,1),'c.');
plot(M_all_tips(:,2),M_all_tips(:,1),'r.','MarkerSize',30);
plot(Minitial_all_tips(counter,2),Minitial_all_tips(counter,1),'ro','MarkerSize',30);
wing = 250;
upper = Minitial_all_tips(counter,1)-wing;
lower = Minitial_all_tips(counter,1)+wing;
left = Minitial_all_tips(counter,2)-wing;
right = Minitial_all_tips(counter,2)+wing;
if upper<=0
    upper = 1;
end
if left<=0
    left = 1;
end
if lower>size(M_OriginMargin,1)
    lower = size(M_OriginMargin,1);
end
if right>size(M_OriginMargin,2)
    right = size(M_OriginMargin,2);
end
axis([left  right  upper lower]);
display(['Current Correction: Tip ',num2str(counter),'/',num2str(size(Minitial_all_tips,1)),'.']);


% --- Executes on button press in RefreshBtn.
function RefreshBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RefreshBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global M_asf;
global M_OriginMargin;
global M_all_pts;
global M_all_tips;
global Minitial_all_tips;
global counter;

close(figure(1));
figure(1);
imshow(M_OriginMargin);hold on;axis off;
plot(M_all_pts(:,2),M_all_pts(:,1),'c.');
plot(M_all_tips(:,2),M_all_tips(:,1),'r.','MarkerSize',30);
plot(Minitial_all_tips(counter,2),Minitial_all_tips(counter,1),'ro','MarkerSize',30);
wing = 250;
upper = Minitial_all_tips(counter,1)-wing;
lower = Minitial_all_tips(counter,1)+wing;
left = Minitial_all_tips(counter,2)-wing;
right = Minitial_all_tips(counter,2)+wing;
if upper<=0
    upper = 1;
end
if left<=0
    left = 1;
end
if lower>size(M_OriginMargin,1)
    lower = size(M_OriginMargin,1);
end
if right>size(M_OriginMargin,2)
    right = size(M_OriginMargin,2);
end
axis([left  right  upper lower]);
display(['Current Correction: Tip ',num2str(counter),'/',num2str(size(Minitial_all_tips,1)),'.']);

