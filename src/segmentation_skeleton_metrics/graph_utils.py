# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""
from random import sample

import networkx as nx


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


def delete_nodes(graph, delete_id, return_cnt=False):
    """
    Deletes nodes in "graph" whose pred_id is identical to "delete_id".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched and edited.
    delete_id : int
        pred_id to be removed from graph.
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
        label = graph.nodes[i]["pred_id"]
        if label == delete_id:
            delete_nodes.append(i)

    # Update graph
    graph.remove_nodes_from(delete_nodes)
    if return_cnt:
        return graph, len(delete_nodes)
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
        graph.nodes[i].update({"pred_id": label})
    return graph


def store_labels(graph):
    """
    Iterates over all nodes in "graph" and stores their pred_id in a graph
    attribute called pred_ids.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be updated

    Returns
    -------
    graph : networkx.Graph
        Updated graph.

    """
    for i in graph.nodes:
        label = graph.nodes[i]["pred_id"]
        graph.graph["pred_ids"].add(label)
    return graph


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


def empty_copy(graph):
    """
    Creates a copy of "graph" that does not contain the node level attributes.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be copied.

    Returns
    -------
    graph : netowrkx.Graph
        Copy of "graph" that does not contain its node level attributes.
    """
    graph_copy = nx.Graph(graph, pred_ids=set())
    for i in graph_copy.nodes():
        graph_copy.nodes[i].clear()
    return graph_copy


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
