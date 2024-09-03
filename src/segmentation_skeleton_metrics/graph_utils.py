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
from scipy.spatial.distance import euclidean as get_dist

from segmentation_skeleton_metrics import utils


# --- Update graph structure ---
def delete_nodes(graph, delete_label, return_cnt=False):
    """
    Deletes nodes in "graph" whose label is identical to "delete_label".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched and edited.
    delete_label : int
        Label to be removed from graph.
    return : bool, optional
        Indication of whether to return the number of nodes deleted from
        "graph". The default is False.

    Returns
    -------
    graph : networkx.Graph
        Updated graph.

    """
    # Find nodes matching delete_marker
    delete_nodes = []
    for i in graph.nodes:
        label = graph.nodes[i]["label"]
        if label == delete_label:
            delete_nodes.append(i)

    # Count deleted edges (if applicable)
    if return_cnt:
        subgraph = graph.subgraph(delete_nodes)
        cnt = subgraph.number_of_edges()

    # Update graph
    graph.remove_nodes_from(delete_nodes)
    if return_cnt:
        return graph, cnt
    else:
        return graph


# -- Label Nodes --
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


def init_label_to_nodes(graph, filter_bool=False, key=None):
    """
    Initializes a dictionary that maps a label to nodes with that label.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be updated
    filter_bool : bool, optional
        Indication of whether to filter labels that occur less frequently than
        a predefined minimum count (MIN_CNT). The default is False.
    key : str
        Graph ID of "graph".

    Returns
    -------
    dict
        Dictionary that maps a label to nodes with that label.

    """
    # Initialize dictionary
    label_to_nodes = defaultdict(set)
    node_to_label = nx.get_node_attributes(graph, "label")
    for i, label in node_to_label.items():
        label_to_nodes[label].add(i)

    # Filter labels (if applicable)
    if filter_bool:
        label_to_nodes = utils.filter_dict(label_to_nodes)

    # Finish
    if key:
        return key, label_to_nodes
    else:
        return label_to_nodes


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
        xyz_1 = utils.to_world(graph.nodes[i]["xyz"])
        xyz_2 = utils.to_world(graph.nodes[j]["xyz"])
        path_length += get_dist(xyz_1, xyz_2)
    return path_length


# -- miscellaneous --
def to_xyz_array(graph):
    """
    Converts node coordinates from a graph into a NumPy array.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that contains nodes with "xyz" attributes.

    Returns
    -------
    numpy.ndarray
        Array where each row represents the 3D coordinates of a node.

    """
    xyz_coords = nx.get_node_attributes(graph, "xyz")
    return np.array([xyz_coords[i] for i in graph.nodes])


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
