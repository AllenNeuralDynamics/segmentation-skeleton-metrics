# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from random import sample


def sample_node(graph):
    """
    Samples a random node from "graph"

    Parameters
    ----------
    graph : networkx.Graph
        Graph in which node is sampled from.

    Returns
    -------
    int
        Node contained in graph.

    """
    return sample(list(graph.nodes), 1)[0]


def get_nbs(graph, i):
    """
    Gets neighbors of node "i".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a neuron.
    i : int
        Node of "graph".

    Returns
    -------
    list[int]
        List of neighbors of node "i".

    """
    return list(graph.neighbors(i))


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
