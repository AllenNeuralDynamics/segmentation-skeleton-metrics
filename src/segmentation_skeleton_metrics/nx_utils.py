# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import networkx as nx
from random import sample


def sample_node(graph):
    return sample(list(graph.nodes), 1)[0]


def get_nbs(graph, i):
    """
    Gets neighbors of node "i".

    Parameters
    ----------
    graph : networkx.Graph()
        Graph that represents a neuron.
    i : int
        Node of "graph".

    Returns
    -------
    list[int]
        List of neighbors of node "i".

    """
    return list(graph.neighbors(i))


def get_labels(labels, graph, edge):
    """
    Gets labels of both nodes contained in "edge" from segmentation
    (i.e. "labels").

    Parameters
    ----------
    labels : dict or np.array
        segmentation.
    graph : networkx.Graph
        Graph that represents a neuron.
    edge : tuple
        Edge in "graph".

    Returns
    -------
    int, int
        Labels of both nodes contained in "edge".

    """
    label_0 = get_label(labels, graph, edge[0])
    label_1 = get_label(labels, graph, edge[1])
    return label_0, label_1


def get_xyz(graph, i):
    """
    Gets (x,y,z) coordinates of node "i".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a neuron.
    i : int
        Node of "graph".

    Returns
    -------
    tuple
        The (x,y,z) coordinates of node "i".

    """
    return tuple(graph.nodes[i]["xyz"])


def get_edge_xyz(graph, edge):
    """
    Gets (x,y,z) coordinates of both nodes contained in "edge".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a neuron.
    edge : tuple
        Edge contained in "graph".

    Returns
    -------
    tuple
        The (x,y,z) coordinates of both nodes contained in "edge".

    """
    return get_xyz(graph, edge[0]), get_xyz(graph, edge[1])


def get_num_edges(list_of_graphs):
    """
    Gets total number of edges of all graphs in "list_of_graphs".

    Parameters
    ----------
    list_of_graphs : list[networkx.Graph]
        DESCRIPTION.

    Returns
    -------
    num_edges : int
        Total number of edges of all graphs in "list_of_graphs".

    """
    num_edges = 0
    for graph in list_of_graphs:
        num_edges += graph.number_of_edges()
    return num_edges
