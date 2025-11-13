% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function all_tips = biDirPairing(all_tips)
% ****** structure of all_tips so far ******   #: Number
%           colume1            colume2           colume3         colume4         colume5            colume6        colume7          colume8          colume9               colume10                colume11          ...
% tip1     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip2     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip3     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% tip4     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center   global index      # of Lives     index of partner1     index of partner2       index of partner3     ...
% ......
% ****** structure of all_tips so far ******   #: Number

for i = 1:size(all_tips,1)
    tip = all_tips(i,:);
    partnerlist = all_tips(i,9:end);
    OriLength = length(partnerlist);
    partnerlist(find(partnerlist==0)) = [];
    tiplife = length(partnerlist);
    
    for j = 1:length(partnerlist)
        partner = all_tips(partnerlist(j),:);
        partnerslist = partner(9:end);
        partnerslist(find(partnerslist==0)) = [];
        if isempty(find(partnerslist==i))
            partnerlist(j) = 0;
            if tiplife>1
                tiplife = tiplife - 1;
            end
        end
    end
    partnerlist(partnerlist==0) = [];
    tipnew = zeros(1,OriLength);
    tipnew(1:length(partnerlist)) = partnerlist;
    all_tips(i,9:end) = tipnew;
    all_tips(i,8) = tiplife;
end



