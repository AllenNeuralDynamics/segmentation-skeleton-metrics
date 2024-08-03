"""
Created on Wed April 8 20:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org



"""

import numpy as np
from scipy.spatial.distance import euclidean as get_dist


def find_sites(graphs, get_labels):
    """
    Detects merges between ground truth graphs which are considered to be
    potential merge sites.

    Parameters
    ----------
    graphs : dict
        Dictionary where the keys are graph ids and values are graphs.
    get_labels : func
        Gets the label of a node in "graphs".

    Returns
    -------
    merge_ids : set[tuple]
        Set of tuples containing a tuple of graph ids and common label between
        the graphs.

    """
    merge_ids = set()
    visited = set()
    for key_1 in graphs.keys():
        for key_2 in graphs.keys():
            keys = frozenset((key_1, key_2))
            if key_1 != key_2 and keys not in visited:
                visited.add(keys)
                intersection = get_labels(key_1).intersection(
                    get_labels(key_2)
                )
                for label in intersection:
                    merge_ids.add((keys, label))
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
        Tuple containing keys corresponding to "graph_1" and "graph_2" along
        the common label between them.

    Returns
    -------
    xyz_pair : list[numpy.ndarray]
        Closest pair of xyz coordinates from "merged_1" and "merged_2".
    min_dist : float
        Distance between xyz coordinates in "xyz_pair".

    """
    min_dist = np.inf
    xyz_pair = list()
    for i in merged_1:
        for j in merged_2:
            xyz_i = graph_1.nodes[i]["xyz"]
            xyz_j = graph_2.nodes[j]["xyz"]
            if get_dist(xyz_i, xyz_j) < min_dist:
                min_dist = get_dist(xyz_i, xyz_j)
                xyz_pair = [xyz_i, xyz_j]
                if min_dist < dist_threshold:
                    print("Merge Detected:", merge_id, xyz_pair, min_dist)
                    return merge_id, xyz_pair, min_dist
    return xyz_pair, min_dist
