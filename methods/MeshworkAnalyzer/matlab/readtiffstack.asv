% convert from meshnetwork_readtiffstack.pro
% A function to read in a tiff stack into a single variable
function imgarr = readtiffstack(imagefile, rect, start_frame, stop_frame)
% imagefile: path to a tiff file
% rect: optional rectangular region to crop from each frame
% [x,y,width,height] measured in pixels from lower left corner (rh coord
% sys)
% start_frame: first frame index to read
% stop_frame: last frame index to read

if nargin < 2
    rect = [];
end

% Read basic TIFF info
info = imfinfo(imagefile);
numImages = numel(info);

if nargin < 3 || isempty(start_frame)
    start_frame = 1;
end
if nargin < 4 || isempty(stop_frame)
    stop_frame = numImages;
end

% Basic sanity check
start_frame = max(1, start_frame);
stop_frame  = min(numImages, stop_frame);
if start_frame > stop_frame
    error('readtiffstack:InvalidRange', ...
        'START_FRAME must be <= STOP_FRAME.');
end

% ---- Determine cropping & preallocate output -------------------------
if isempty(rect)
    % No cropping: read first frame to get size and type
    firstSlice = imread(imagefile, start_frame);
    [rows, cols] = size(firstSlice);
    imgarr = zeros(rows, cols, stop_frame - start_frame + 1, ...
        class(firstSlice));
else
    % RECT = [x y width height] from *lower-left*, IDL-style
    x      = rect(1);
    y      = rect(2);
    width  = rect(3);
    height = rect(4);

    % Convert from lower-left coords to MATLAB's upper-left coords
    imgHeight = info(1).Height;
    % bottom and top in IDL coordinates (0-based)
    y_bottom = y;
    y_top    = y + height - 1;

    % Row indices in MATLAB (1-based, top=1)
    row_start = imgHeight - y_top;
    row_end   = imgHeight - y_bottom;

    % Column indices in MATLAB (1-based, left=1)
    col_start = x + 1;
    col_end   = x + width;

    % Read first cropped slice to get type
    firstSlice = imread(imagefile, start_frame, ...
        'PixelRegion', { [row_start row_end], ...
        [col_start col_end] });
    [rows, cols] = size(firstSlice);
    imgarr = zeros(rows, cols, stop_frame - start_frame + 1, ...
        class(firstSlice));
end

% ---- Read frames into the stack -------------------------------------
idx = 1;
for k = start_frame:stop_frame
    if isempty(rect)
        img = imread(imagefile, k);
    else
        img = imread(imagefile, k, ...
            'PixelRegion', { [row_start row_end], ...
            [col_start col_end] });
    end

    % If the TIFF is unexpectedly multi-channel, you might need to
    % adapt this to store a 4-D array; here we assume grayscale.
    if ndims(img) ~= 2
        error('readtiffstack:NotGrayscale', ...
            'Frame %d is not a 2-D grayscale image.', k);
    end

    imgarr(:, :, idx) = img;
    idx = idx + 1;
end
end