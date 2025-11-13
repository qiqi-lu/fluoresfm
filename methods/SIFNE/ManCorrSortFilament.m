% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function AnalysisInfo= ManCorrSortFilament(filament)

filament(find(filament(:,1)==0),:) = [];
d = 0;
tips = [filament(1,:); filament(end,:)];
changing = tips(1,:);
temp = filament;

while ~isempty(temp)
    k = dsearchn(temp,changing);
    d = d + pdist([changing; temp(k,:)]);
    % update information
    changing = temp(k,:);
    temp(k,:) = [];
end
load data\L;
maskI = zeros(size(L));
maskI(sub2ind(size(L),filament(:,1),filament(:,2))) = 1;
orient = regionprops(maskI,'Orientation');
orient = orient(1).Orientation;

TotalLength = d;
EndToEndDist = pdist([tips(1,1),tips(1,2);tips(2,1),tips(2,2)]);

AnalysisInfo = [orient  TotalLength  EndToEndDist];