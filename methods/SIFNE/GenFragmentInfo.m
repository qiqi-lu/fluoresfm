% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function FragInfo = GenFragmentInfo(L,i,all_tips,FragInfo)

idx = find(all_tips(:,3)==i);
curr_tip = all_tips(idx(1),1:2);
[x y] = find(L==i);
ptslist = zeros(1,2*length(x));
ptslist((1:length(x))*2-1) = x;
ptslist((1:length(x))*2) = y;
FragInfo(1) = i;
FragInfo(2) = length(x);
FragInfo(3) = all_tips(idx(1),1);
FragInfo(4) = all_tips(idx(1),2);
FragInfo(5) = all_tips(idx(2),1);
FragInfo(6) = all_tips(idx(2),2);
FragInfo(7:6+length(ptslist)) = ptslist;

