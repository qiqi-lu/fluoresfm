% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function flag =  LargeCurFlag(x, y) % coordinates consistent with image in Matlab
% normalize coordinates, shift coordinates so the smallest x and y start
% from 1
x = x-min(x)+1;
y = y-min(y)+1;

% create a local binary patch that exactly fits around the fragment with a
% 1 pixel margin
temp = zeros(max(x)-min(x)+3, max(y)-min(y)+3);
x = x+1;
y = y+1;
temp(sub2ind(size(temp),x,y)) = 1;

% count neighboring pixels for each fragment pixel
s1 = temp(sub2ind(size(temp),x-1,y-1));
s2 = temp(sub2ind(size(temp),x-1,y));
s3 = temp(sub2ind(size(temp),x-1,y+1));
s4 = temp(sub2ind(size(temp),x,y-1));
s5 = temp(sub2ind(size(temp),x,y));
s6 = temp(sub2ind(size(temp),x,y+1));
s7 = temp(sub2ind(size(temp),x+1,y-1));
s8 = temp(sub2ind(size(temp),x+1,y));
s9 = temp(sub2ind(size(temp),x+1,y+1));

s = s1+s2+s3+s4+s5+s6+s7+s8+s9;
tipidx = find(s==2); % endpoints (tips) has exactly 2 ones, middle pixel has 3.

if length(tipidx)~=2
    % if the fragment doesn't have exactly two endpoints, it's likely a
    % loop, branch or broken curve, so make it as invalid
    flag = 1;
else
    % measure curvature
    x_change = x;
    y_change = y;
    tipchange = [x(tipidx(1))  y(tipidx(1))]; % start at one endpoint
    d = 0;
    for i = 1:length(x)
        k = dsearchn([x_change  y_change],tipchange); % find the nearest remaining pixel
        d = d + pdist([tipchange; [x_change(k)  y_change(k)]]); % accumulates the Euclidean distance.
        % d is about the curve length of the fragment
        
        tipchange = [x_change(k)  y_change(k)]; % move to that point
        x_change(k) = []; % remove visited pixels
        y_change(k) = [];
    end
    
    % compare curved length vs. straight distance
    if 1.2 * pdist([[x(tipidx(1))  y(tipidx(1))];[x(tipidx(2))  y(tipidx(2))]])<d
        flag = 1;
    else
        flag = 0;
    end
    
end


