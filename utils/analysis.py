"""
Functions used to analysis images, such as structure morphology.
"""

import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_dilation
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max, blob_log
from scipy.ndimage import distance_transform_edt
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.networking import Network
from nellie.feature_extraction.hierarchical import Hierarchy
from utils.data import normalization
from skimage.draw import circle_perimeter


def node_degree(im_info, verbose=True):
    """
    Calculate the node degree of ER network using Nellie.
    For 2D image with a shape of [T, Y, X].
    ### Parameters:
    - `im_info`: ImInfo object.
    - `verbose`: bool, whether to print the progress.
    ### Returns:
    - `node_info`: dict, containing the node coordinates and node degree.
        ["coords", "degree", "pixel_class"]
    """
    # --------------------------------------------------------------------------
    if verbose:
        print("-" * 80)
        print("[INFO] ER network analysis ...")
    preprocessing = Filter(im_info, remove_edges=False)
    preprocessing.run()
    segmentation = Label(im_info, otsu_thresh_intensity=False, threshold=None)
    segmentation.run()
    network_analyzer = Network(im_info, num_t=1)
    network_analyzer.run()

    # --------------------------------------------------------------------------
    # get the skeleton
    # skel = im_info.get_memmap(im_info.pipeline_paths["im_skel"])[0]
    pixel_class = im_info.get_memmap(im_info.pipeline_paths["im_pixel_class"])[0]

    junctions = pixel_class == 4
    junction_labels = label(junctions)
    junction_objects = regionprops(junction_labels)

    # get juction centroids
    node_coords = []
    for obj in junction_objects:
        node_coords.append(obj.centroid)
    node_coords = np.array(node_coords)
    if verbose:
        print("[INFO] Number of node centroids: ", node_coords.shape[0])

    # --------------------------------------------------------------------------
    # calclulate the node degree
    node_degree = []
    node_label = []

    strcture = np.ones((3, 3), dtype=bool)  # 2D connectively structure

    for i_node, node in enumerate(node_coords):
        node_label.append(junction_objects[i_node].label)

        y, x = map(int, np.round(node))
        neighborhood_size = 3

        y_min = max(0, y - neighborhood_size)
        y_max = min(pixel_class.shape[0], y + neighborhood_size + 1)
        x_min = max(0, x - neighborhood_size)
        x_max = min(pixel_class.shape[1], x + neighborhood_size + 1)

        # extract neighborhood
        neighborbood = pixel_class[y_min:y_max, x_min:x_max]
        # get the node position in the neighborhood
        offset_y, offset_x = 0, 0
        if y < neighborhood_size:
            offset_y = y - neighborhood_size
        if x < neighborhood_size:
            offset_x = x - neighborhood_size
        node_pos_roi = (neighborhood_size + offset_y, neighborhood_size + offset_x)

        # find all branch pixels (3) and endpoint pixels (2) in the neighborhood
        branch_endpoint_pixels = (neighborbood == 3) | (neighborbood == 2)
        # label connceted components of branch and endpoint pixels
        branch_endpoint_labels = label(branch_endpoint_pixels, connectivity=2)

        # find all the junction pixels in the neighborhood
        junction_pixels = neighborbood == 4

        unique_labels = np.unique(branch_endpoint_labels)
        unique_labels = unique_labels[unique_labels != 0]  # exclude background

        # check each unique branck label, check it it connects to the junction
        connected_branch = set()
        for lable in unique_labels:
            branch_mask = branch_endpoint_labels == lable
            dialated_junction_mask = binary_dilation(
                junction_pixels, footprint=strcture
            )
            if np.any(branch_mask & dialated_junction_mask):
                connected_branch.add(lable)

        # calculate the node degree
        node_degree.append(len(connected_branch))
    node_degree = np.array(node_degree)
    node_info = {
        "coords": node_coords,
        "degree": node_degree,
        "pixel_class": pixel_class,
    }
    return node_info


def pit_segmentation(
    image,
    gaussian_sigma=0.0,
    norm_range=(0.03, 0.995),
    clip_range=(0, 0.9),
    min_area_px=3,
    hole_area_px=8,
    min_peak_distance_px=3,
    return_intermediate=False,
    otsu_thr_factor=0.25,
):
    """
    Calculate the diamter of pits in the image and count the number of them.
    ### Parameters:
    - `image`: 2D numpy array.
    """

    assert image.ndim == 2, "[ERROR] Image must be 2D."

    # --------------------------------------------------------------------------
    # segmentation
    # --------------------------------------------------------------------------
    # Gaussian smoothing
    if gaussian_sigma > 0:
        smoothed = gaussian(image, sigma=gaussian_sigma)
    else:
        smoothed = image

    # Normalize to 0-1
    normalizer = lambda x: normalization(x, p_low=norm_range[0], p_high=norm_range[1])
    normalized = np.clip(normalizer(smoothed), clip_range[0], clip_range[1])

    # Thresholding
    thr = threshold_otsu(normalized)
    mask_ostu = normalized > thr * otsu_thr_factor
    # Morphological operations
    mask_hole_fill = remove_small_holes(mask_ostu, area_threshold=hole_area_px)
    # remove small objects
    mask_init = remove_small_objects(mask_hole_fill, min_size=min_area_px)

    # watershed segmentation ---------------------------------------------------
    # generate the markers a local maxima of the distance to the background
    dist_map = distance_transform_edt(mask_init)
    peak_coords = peak_local_max(
        dist_map, labels=mask_init, min_distance=min_peak_distance_px
    )
    markers = np.zeros(dist_map.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = i

    # Watershed on negative distance splits blobs at saddle points
    labels_ws = watershed(-dist_map, markers, mask=mask_init)
    labels_ws_clean = remove_small_objects(labels_ws, min_size=min_area_px)

    if return_intermediate:
        return {
            "image": image,
            "smoothed": smoothed,
            "normalized": normalized,
            "mask_ostu": mask_ostu,
            "mask_hole_fill": mask_hole_fill,
            "mask_init": mask_init,
            "dist_map": dist_map,
            "peak_coords": peak_coords,
            "markers": markers,
            "labels_ws": labels_ws,
            "labels_ws_clean": labels_ws_clean,
        }
    else:
        return labels_ws_clean


from skimage import draw
from skimage.feature import blob_doh, blob_dog


def lysosome_segmentation(
    image,
    gaussian_sigma=0.0,
    norm_range=(0.03, 0.995),
    clip_range=(0, 0.9),
):
    """
    lysosome_segmentation.
    """

    assert image.ndim == 2, "[ERROR] Image must be 2D."
    image_shape = image.shape
    # --------------------------------------------------------------------------
    # segmentation
    # --------------------------------------------------------------------------
    # Gaussian smoothing
    if gaussian_sigma > 0:
        smoothed = gaussian(image, sigma=gaussian_sigma)
    else:
        smoothed = image

    # Normalize to 0-1
    normalizer = lambda x: normalization(x, p_low=norm_range[0], p_high=norm_range[1])
    normalized = np.clip(normalizer(smoothed), clip_range[0], clip_range[1])

    # use blob_log to detect the lysosome
    # blobs = blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)
    blobs = blob_dog(image, max_sigma=30, threshold=0.1)
    # blobs = blob_doh(image, max_sigma=30, threshold=0.01)
    # compute radii in the 3rd column
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    # convert blobs into segmentation labels
    labels = np.zeros_like(image, dtype=np.int32)
    for i, (y, x, r) in enumerate(blobs):
        y_int, x_int = map(int, np.round([y, x]))
        radius = int(np.round(r))
        rr, cc = draw.disk((y_int, x_int), radius, shape=image_shape)
        labels[rr, cc] = i + 1

    return labels
