"""
Functions used to analysis images, such as structure morphology.
"""

from nellie.segmentation.networking import Network

from nellie.feature_extraction.hierarchical import Hierarchy
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
import numpy as np


def node_degree(im_info, verbose=True):
    """
    Calculate the node degree of ER network using Nellie.
    For 2D image with a shape of [T, Y, X].
    ### Parameters:
    - `im_info`: ImInfo object.
    - `verbose`: bool, whether to print the progress.
    ### Returns:
    - `node_info`: dict, containing the node coordinates and node degree.
    """
    # --------------------------------------------------------------------------
    if verbose:
        print("-" * 80)
        print("[INFO] ER network analysis ...")
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
    }
    return node_info
