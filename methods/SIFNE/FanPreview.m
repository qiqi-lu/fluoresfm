% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function FanEdge = FanPreview(fanAngle, L, r, base_ori, base_loc)
%  global index in label list/list of all tips

FanPoints = CreateFan(r, base_ori, fanAngle);

% used to generate fan edge below
FanPoints1 = [FanPoints(:,1)+base_loc(1),  FanPoints(:,2)+base_loc(2)];
FanPoints1(find(FanPoints1(:,1)<=0),:) = []; % make sure that Fan Points are not out of boundary
FanPoints1(find(FanPoints1(:,2)<=0),:) = []; % make sure that Fan Points are not out of boundary
FanPoints1(find(FanPoints1(:,1)>size(L,1)),:) = []; % make sure that Fan Points are not out of boundary
FanPoints1(find(FanPoints1(:,2)>size(L,2)),:) = []; % make sure that Fan Points are not out of boundary
FanTemp = zeros(size(L));
FanTemp (sub2ind(size(FanTemp), FanPoints1(:,1), FanPoints1(:,2))) = 1;
FanEdge = bwboundaries(im2bw(FanTemp));
FanEdge = FanEdge{1};
% used to generate fan edge above


