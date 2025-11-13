% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function FanPoints = CreateFan(radius, base_ori, ori_width)

imageSize = radius*4+1;
[columnsInImage rowsInImage] = meshgrid(1:imageSize, 1:imageSize);
centerX = radius*2 + 1;
centerY = centerX;
circlePixels = ((rowsInImage - centerY).^2  + (columnsInImage - centerX).^2) <= radius.^2;

angle1 = (base_ori - ori_width/2)*pi/180;
angle2 = (base_ori + ori_width/2)*pi/180;

netcorner1 = [(radius*2+1)*cos(angle1),   (radius*2+1)*sin(angle1)];
netcorner2 = [(radius*2+1)*cos(angle2),   (radius*2+1)*sin(angle2)];

corner1 = [centerX-netcorner1(2)    centerY+netcorner1(1)];
corner2 = [centerX-netcorner2(2)    centerY+netcorner2(1)];

mask = zeros(imageSize);
mask = roipoly(mask, [centerY,corner1(2),corner2(2)],  [centerX,corner1(1),corner2(1)]);
fan = and(circlePixels,mask);

[x y] = find(fan==1);
x = x - centerX;
y = y - centerY;
FanPoints = [x y];
end