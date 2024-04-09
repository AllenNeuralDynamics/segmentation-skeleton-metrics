# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""
from random import sample

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean as dist


# -- edit graph --
def remove_edge(graph, i, j):
    """
    Remove the edge "(i,j)" from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to edited.
    i : int
        Node part of edge to be removed.
    j : int
        Node part of edge to be removed.

    Returns
    -------
    graph : networkx.Graph
        Graph with edge removed.

    """
    graph.remove_edges_from([(i, j)])
    return graph


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


# -- attribute utils --
def get_coord(graph, i):
    """
    Gets (x,y,z) image coordinates of node "i".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be queried.
    i : int
        Node of "graph".

    Returns
    -------
    tuple
        The (x,y,z) image coordinates of node "i".

    """
    return tuple(graph.nodes[i]["xyz"])


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
    graph : networkx.Graph
        Updated graph.

    """
    for i in nodes:
        graph.nodes[i].update({"label": label})
    return graph


def store_labels(graph):
    """
    Iterates over all nodes in "graph" and stores the label and node id in
    a dictionary called "label_to_node".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be updated

    Returns
    -------
    label_to_node : dict
        Dictionary that stores the label and node id.

    """
    label_to_node = dict()
    for i in graph.nodes:
        label = graph.nodes[i]["label"]
        if label in label_to_node.keys():
            label_to_node[label].add(i)
        else:
            label_to_node[label] = set([i])
    return label_to_node


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
            run_lengths.append(compute_path_length(subgraph))
    else:
        run_lengths.append(0)
    return np.array(run_lengths)


def compute_path_length(graph):
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
        xyz_1 = graph.nodes[i]["xyz"]
        xyz_2 = graph.nodes[j]["xyz"]
        path_length += dist(xyz_1, xyz_2)
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


def sample_node(graph):
    """
    Samples a node from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be sampled from.

    Returns
    -------
    int
        Node.

    """
    return sample(list(graph.nodes), 1)[0]
