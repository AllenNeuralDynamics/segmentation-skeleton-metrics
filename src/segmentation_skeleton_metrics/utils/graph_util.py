# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import sample

import networkx as nx
import numpy as np
from scipy.spatial import distance

from segmentation_skeleton_metrics.utils import img_util

ANISOTROPY = np.array([0.748, 0.748, 1.0])


def to_graph(swc_dict):
    """
    Builds a graph from a dictionary that contains the contents of an SWC
    file.

    Parameters
    ----------
    swc_dict : dict
        ...

    Returns
    -------
    networkx.Graph
        Graph built from an SWC file.

    """
    # Initializations
    old_to_new = dict()
    run_length = 0
    voxels = np.zeros((len(swc_dict["id"]), 3), dtype=np.int32)

    # Build graph
    graph = nx.Graph()
    for i in range(len(swc_dict["id"])):            
        # Get node id
        old_id = swc_dict["id"][i]
        old_to_new[old_id] = i

        # Update graph
        voxels[i] = swc_dict["voxel"][i]
        if swc_dict["pid"][i] != -1:
            # Add edge
            parent = old_to_new[swc_dict["pid"][i]]
            graph.add_edge(i, parent)

            # Update run length
            xyz_i = voxels[i] * ANISOTROPY
            xyz_p = voxels[parent] * ANISOTROPY
            run_length += distance.euclidean(xyz_i, xyz_p)

    # Set graph-level attributes
    graph.graph["n_edges"] = graph.number_of_edges()
    graph.graph["run_length"] = run_length
    graph.graph["voxel"] = voxels
    return {swc_dict["swc_id"]: graph}


# --- Update graph ---
def delete_nodes(graph, target_label):
    """
    Deletes nodes in "graph" whose label is "target_label".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched and edited.
    target_label : int
        Label to be deleted from graph.

    Returns
    -------
    networkx.Graph
        Updated graph.

    """
    delete_nodes = []
    for i in graph.nodes:
        label = graph.nodes[i]["label"]
        if label == target_label:
            delete_nodes.append(i)
    graph.remove_nodes_from(delete_nodes)
    return graph


def upd_labels(graph, nodes, label):
    """
    Updates the label of each node in "nodes" with "label".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be updated.
    nodes : list
        List of nodes to be updated.
    label : int
        New label of each node in "nodes".

    Returns
    -------
    networkx.Graph
        Updated graph.

    """
    for i in nodes:
        graph.nodes[i].update({"label": label})
    return graph


def init_label_to_nodes(graph):
    """
    Initializes a dictionary that maps a label to nodes with that label.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    dict
        Dictionary that maps a label to nodes with that label.

    """
    label_to_nodes = defaultdict(set)
    node_to_label = nx.get_node_attributes(graph, "label")
    for i, label in node_to_label.items():
        label_to_nodes[label].add(i)
    return label_to_nodes


# -- eval tools --
def count_splits(graph):
    """
    Counts the number of splits in "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be evaluated.

    Returns
    -------
    int
        Number of splits in "graph".

    """
    return max(len(list(nx.connected_components(graph))) - 1, 0)


def compute_run_lengths(graph):
    """
    Computes the path length of each connected component in "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be parsed.

    Returns
    -------
    run_lengths : numpy.ndarray
        Array containing run lengths of each connected component in "graph".

    """
    run_lengths = []
    if graph.number_of_nodes():
        for nodes in nx.connected_components(graph):
            subgraph = graph.subgraph(nodes)
            run_lengths.append(compute_run_length(subgraph))
    else:
        run_lengths.append(0)
    return np.array(run_lengths)


def compute_run_length(graph):
    """
    Computes path length of graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be parsed.

    Returns
    -------
    path_length : float
        Path length of graph.

    """
    path_length = 0
    for i, j in nx.dfs_edges(graph):
        xyz_i = img_util.to_physical(graph.graph["voxel"][i], ANISOTROPY)
        xyz_j = img_util.to_physical(graph.graph["voxel"][j], ANISOTROPY)
        path_length += distance.euclidean(xyz_i, xyz_j)
    return path_length


# -- miscellaneous --
def sample_leaf(graph):
    """
    Samples leaf node from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be sampled from.

    Returns
    -------
    int
        Leaf node of "graph"

    """
    leafs = [i for i in graph.nodes if graph.degree[i] == 1]
    return sample(leafs, 1)[0]


def get_node_labels(graphs):
    """
    Creates a dictionary that maps a graph id to the set of unique labels of
    nodes in that graph.

    Parameters
    ----------
    graphs : dict
        Graphs to be searched.

    Returns
    -------
    dict
        Dictionary that maps a graph id to the set of unique labels of nodes
        in that graph.

    """
    with ProcessPoolExecutor() as executor:
        # Assign processes
        processes = list()
        for key, graph in graphs.items():
            processes.append(
                executor.submit(init_label_to_nodes, graph, True, key)
            )

        # Store results
        graph_to_labels = dict()
        for cnt, process in enumerate(as_completed(processes)):
            key, label_to_nodes = process.result()
            graph_to_labels[key] = set(label_to_nodes.keys())
    return graph_to_labels
