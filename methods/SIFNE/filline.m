% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function I = filline(I,value,x1,y1,x2,y2)

for x=x1:sign(x2-x1):x2
    I(round(x),round(((y2-y1)/(x2-x1))*(x-x1)+y1))=value;
end

for y=y1:sign(y2-y1):y2
    I(round(((x2-x1)/(y2-y1))*(y-y1)+x1),round(y))=value;
end

end