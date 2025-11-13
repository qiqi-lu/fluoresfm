% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function varargout = LoadImg(varargin)
% LOADIMG MATLAB code for LoadImg.fig
%      LOADIMG, by itself, creates a new LOADIMG or raises the existing
%      singleton*.
%
%      H = LOADIMG returns the handle to a new LOADIMG or the handle to
%      the existing singleton*.
%
%      LOADIMG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LOADIMG.M with the given input arguments.
%
%      LOADIMG('Property','Value',...) creates a new LOADIMG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LoadImg_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LoadImg_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LoadImg

% Last Modified by GUIDE v2.5 17-Mar-2015 11:00:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @LoadImg_OpeningFcn, ...
    'gui_OutputFcn',  @LoadImg_OutputFcn, ...
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


% --- Executes just before LoadImg is made visible.
function LoadImg_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to LoadImg (see VARARGIN)

% Choose default command line output for LoadImg
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes LoadImg wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = LoadImg_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in load_button.
function load_button_Callback(hObject, eventdata, handles)
% hObject    handle to load_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
warning off;
close(figure(1));
imgpath=imgetfile;
mkdir data;
save data\imgpath.mat imgpath;
OriginImg = imread(imgpath);
if length(size(OriginImg))==3
    OriginImg = rgb2gray(OriginImg);
end
OriginImg = imadjust(im2uint8(OriginImg));
save data\OriginImg.mat OriginImg;
close all;
figure('name','Please Check The Image Loaded');
imshow(OriginImg);axis off;
LFT_OFT;
