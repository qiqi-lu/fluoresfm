% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function LinkInfo = GenLinkageInfo(i,L,connects,LinkInfo)
connects(find(connects(:,1)==0),:) = [];
Llist = L(sub2ind(size(L),connects(:,1),connects(:,2)));
Llist = unique(Llist);
LinkInfo(1:(size(connects,1))/2,1) = i;
for j = 1:length(Llist)
    LinkInfo(j,2) = Llist(j);
    [x y] = find(L==Llist(j));
    LinkInfo(j,(1:length(x))*2+1) = x;
    LinkInfo(j,(1:length(x))*2+2) = y;
end
