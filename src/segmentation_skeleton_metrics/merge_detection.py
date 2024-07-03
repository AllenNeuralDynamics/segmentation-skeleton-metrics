"""
Created on Wed April 8 20:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org



"""

import numpy as np

from segmentation_skeleton_metrics import utils


def detect_potentials(labeled_graphs, get_labels):
    """
    Detects merges between ground truth graphs which are considered to be
    potential merge sites.

    Parameters
    ----------
    labeled_graphs : dict
        Dictionary where the keys are graph ids and the values are the
        corresponding graphs.
    get_labels : func
        Gets the label of a node in "labeled_graphs".

    Returns
    -------
    merge_ids : set[tuple]
        Set of tuples containing a tuple of graph ids and common label between
        the graphs.

    """
    merge_ids = set()
    visited = set()
    for id_1 in labeled_graphs.keys():
        for id_2 in labeled_graphs.keys():
            ids = frozenset((id_1, id_2))
            if id_1 != id_2 and ids not in visited:
                visited.add(ids)
                intersection = get_labels(id_1).intersection(get_labels(id_2))
                for label in intersection:
                    merge_ids.add((ids, label))
    return merge_ids


def localize(graph_1, graph_2, merged_1, merged_2, dist_threshold, merge_id):
    """
    Finds the closest pair of xyz coordinates from "merged_1" and "merged_2".

    Parameters
    ----------
    graph_1 : networkx.Graph
        Graph with potential merge.
    graph_2 : networkx.Graph
        Graph with potential merge.
    merged_1 : set
        Nodes contained in "graph_1" with same labels as nodes in "merged_2".
    merged_2 : set
        Nodes contained in "graph_2" with same labels as nodes in "merged_1".
    dist_threshold : float
        Distance that determines whether two graphs contain a merge site.
    merge_id : tuple
        Tuple containing ids corresponding to "graph_1" and "graph_2" along
        the common label between them.

    Returns
    -------
    xyz_pair : list[numpy.ndarray]
        Closest pair of xyz coordinates from "merged_1" and "merged_2".
    min_dist : float
        Distance between xyz coordinates in "xyz_pair".

    """
    # Check whether merge is spurious
    if len(merged_1) < 16 and len(merged_2) < 16:
        return merge_id, None, np.inf

    # Compute pairwise distances
    min_dist = np.inf
    xyz_pair = list()
    for i in merged_1:
        for j in merged_2:
            xyz_i = graph_1.nodes[i]["xyz"]
            xyz_j = graph_2.nodes[j]["xyz"]
            if utils.dist(xyz_i, xyz_j) < min_dist:
                min_dist = utils.dist(xyz_i, xyz_j)
                xyz_pair = [xyz_i, xyz_j]
                if min_dist < dist_threshold:
                    return merge_id, xyz_pair, min_dist
    return merge_id, xyz_pair, min_dist

"""
def intersections(get_projection, xyz_list, xyz_to_id, close_dist_threshold):
    Projects coordinates in "xyz_list" onto ground truth graphs, then stores
    the correspond label in "hit_ids" if the projection distance is less than
    "dist_threshold".

    Parameters
    ----------
    get_projection : func
        Computes projections onto ground truth graphs.
    xyz_list : list
        List of xyz coordinates to be projected.
    xyz_to_id : dict
        Dictionary where keys are xyz coordinates in the ground truth graphs
        and values are a dictionary that stores the corresonding label and
        node ids.
    close_dist_threshold : float
        Threshold that defines whether a projection distance is small enough
        to be considered a valid intersection.

    Returns
    -------
    hit_ids : dict
        Dictionary where deys are labels and values are node ids.

    hit_ids = dict()
    multi_hits = set()
    for xyz in xyz_list:
        hat_xyz, d = get_projection(xyz)
        if d < close_dist_threshold:
            hits = list(xyz_to_id[hat_xyz].keys())
            if len(hits) > 1:
                multi_hits.add(hat_xyz)
            else:
                hat_i = xyz_to_id[hat_xyz][hits[0]]
                hit_ids = utils.append_dict_value(hit_ids, hits[0], hat_i)
    return utils.resolve(multi_hits, hit_ids, xyz_to_id)
"""
