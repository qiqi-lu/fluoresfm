% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function [sorted_filament AnalysisInfo]= SortFilament(filament, maskI, ConnectPts, Fullength)

sorted_filament = zeros(Fullength,2);
x = filament(:,1);
y = filament(:,2);
x(find(x==0)) = [];
y(find(y==0)) = [];

maskI(sub2ind(size(maskI),x,y)) = 1;
L = bwlabel(maskI);
ConnectPts(find(ConnectPts(:,1)==0),:) = [];
if size(ConnectPts,1)~=2
    L_uniqlist = [];
    all = [];
    L_list = [];
    for j = 1:(size(ConnectPts,1)/2-1)
        L = filline(L,L(sub2ind(size(L),ConnectPts(j*2,1),ConnectPts(j*2,2))),ConnectPts(j*2,1),ConnectPts(j*2,2),ConnectPts(j*2+1,1),ConnectPts(j*2+1,2));
        [Lx, Ly] = find(L==L(sub2ind(size(L),ConnectPts(j*2,1),ConnectPts(j*2,2))));
        all = [all; Lx Ly];
        L_list = [L_list; L(sub2ind(size(L),ConnectPts(j*2,1),ConnectPts(j*2,2))) * ones(length(Lx),1)];
        L_uniqlist = [L_uniqlist  L(sub2ind(size(L),ConnectPts(j*2,1),ConnectPts(j*2,2)))];
    end
    [Lx, Ly] = find(L==L(sub2ind(size(L),ConnectPts(end,1),ConnectPts(end,2))));
    all = [all; Lx Ly];
    L_list = [L_list; L(sub2ind(size(L),ConnectPts(end,1),ConnectPts(end,2))) * ones(length(Lx),1)];
    L_uniqlist = [L_uniqlist  L(sub2ind(size(L),ConnectPts(end,1),ConnectPts(end,2)))];
else
    L_uniqlist = 1;
    L_list = L(sub2ind(size(L),x,y));
    all = [x y];
end

BW = L;
BW(find(BW~=0))=1;
[x, y] = find(BW==1);
tips = [];
for k = 1:length(x)
    temp = BW(   (x(k)-1):(x(k)+1)   ,   (y(k)-1):(y(k)+1)  );
    if sum(temp(:))==2 && temp(2,2)~=0
        tips = [tips;[x(k)  y(k)]];
    end
end

if L(tips(1,1),tips(1,2))~=L_uniqlist(1)
    all = flipud(all);
    L_list = flipud(L_list);
    L_uniqlist = fliplr(L_uniqlist);
end

d = 0;
SingleFilledSortedFilament = [];

changing = tips(1,:);
for i = 1:length(L_uniqlist)
    temp = all(find(L_list==L_uniqlist(i)),:);
    while ~isempty(temp)
        k = dsearchn(temp,changing);
        d = d + pdist([changing; temp(k,:)]);
        
        % update information
        changing = temp(k,:);
        SingleFilledSortedFilament = [SingleFilledSortedFilament;temp(k,:)];
        temp(k,:) = [];
    end
end
sorted_filament(1:size(SingleFilledSortedFilament,1),1:size(SingleFilledSortedFilament,2)) = SingleFilledSortedFilament;
orient = regionprops(maskI,'Orientation');
ctrs = regionprops(maskI,'Centroid');
ctrs = ctrs(1).Centroid;
orient = orient(1).Orientation;
TotalLength = d;
EndToEndDist = pdist([tips(1,1),tips(1,2);tips(2,1),tips(2,2)]);

AnalysisInfo = [orient  TotalLength  EndToEndDist  ctrs(2)  ctrs(1)];