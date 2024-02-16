# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""
from random import sample

import networkx as nx

from segmentation_skeleton_metrics import utils


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


def get_coord(graph, i):
    """
    Gets (x,y,z) image coordinates of node "i".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a neuron.
    i : int
        Node of "graph".

    Returns
    -------
    tuple
        The (x,y,z) image coordinates of node "i".

    """
    return tuple(graph.nodes[i]["xyz"])


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
