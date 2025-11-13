% normalize image
% clip normalized image

% @qiqilu
function image_norm = normalize_clip(image)
    image = double(image);

    p_low = prctile(image(:), 3);
    p_high = prctile(image(:), 99.5);

    if p_high == p_low
        image_norm = zeros(size(image));
        return;
    end

    image_norm = (image-p_low)./(p_high-p_low);
    image_norm(image_norm<0)=0;
    image_norm(image_norm>1)=1;
end

