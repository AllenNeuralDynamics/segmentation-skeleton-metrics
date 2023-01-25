# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import numpy as np
import tensorstore as ts


# Miscellaneous routines
def get_value(volume, graph, i):
    """
    Gets voxel value of node "i".

    Parameters
    ----------
    volume : dict
        Sparse image volume.
    graph : networkx.Graph
        Graph which represents a neuron.
    i : int
        Node of "graph".

    Returns
    -------
    int
       Voxel value of node "i".

    """
    idx = get_idx(graph, i)
    return volume[idx] if idx in volume.keys() else 0


def mkdir(path_to_dir):
    """
    Makes a directory if it does not already exist.

    Parameters
    ----------
    path_to_dir : str
        Path to directory.

    Returns
    -------
    None.

    """
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


def check_edge(set_of_edges, edge):
    """
    Checks whether "edge" is in "set_of_edges".

    Parameters
    ----------
    set_of_edges : set
        Set of edges.
    edge : tuple
        Edge.

    Returns
    -------
    bool : bool
        Indication of whether "edge" is contained in "set_of_edges".

    """
    edge_rev = list(edge)
    edge_rev.reverse()
    if edge in set_of_edges or edge_rev in set_of_edges:
        return True
    else:
        return False


def remove_edge(set_of_edges, edge):
    """
    Checks whether "edge" is in "set_of_edges" and removes it.

    Parameters
    ----------
    set_of_edges : set
        Set of edges.
    edge : tuple
        Edge.

    Returns
    -------
    set_of_edges : set
        Updated set of edges such that "edge" is removed if it was
        contained in "set_of_edges".

    """
    edge_reverse = list(edge)
    edge_reverse.reverse()
    if edge_reverse in set_of_edges:
        set_of_edges.remove(edge_reverse)
    elif edge in set_of_edges:
        set_of_edges.remove(edge)
    return set_of_edges


def upload_google_pred(path_to_data):
    """
    Uploads segmentation mask stored as a directory of shard files.

    Parameters
    ----------
    path_to_data : str
        Path to directory containing shard files.

    Returns
    -------
    sparse_volume : dict
        Sparse image volume.

    """
    dataset_ts = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "file",
                "path": path_to_data,
            },
        }
    ).result()
    dataset_ts = dataset_ts[ts.d[:].transpose[::-1]]
    volume_ts = dataset_ts[ts.d["channel"][0]]
    sparse_volume = dict()
    for x in range(volume_ts.shape[0]):
        plane_x = np.array(volume_ts[x, :, :].read().result())
        y, z = np.nonzero(plane_x)
        for i in range(len(y)):
            sparse_volume[(x, y[i], z[i])] = plane_x[y[i], z[i]]
    return sparse_volume


# Networkx helper routines
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


def get_edge_values(volume, graph, edge):
    """
    Gets voxel value of both nodes contained in "edge".

    Parameters
    ----------
    volume : dict
        Sparse image volume.
    graph : networkx.Graph
        Graph that represents a neuron.
    edge : tuple
        Edge in "graph".

    Returns
    -------
    tuple
        Voxel values of both nodes contained in "edge".

    """
    return get_value(volume, graph, edge[0]), get_value(volume, graph, edge[1])


def get_idx(graph, i):
    """
    Gets voxel index of node "i".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a neuron.
    i : int
        Node of "graph".

    Returns
    -------
    tuple
        voxel index of node "i".

    """
    return tuple(graph.nodes[i]["idx"])


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
    return graph.nodes[i]["xyz"]


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


def get_edge_idx(graph, edge):
    """
    Gets indices of both nodes contained in "edge".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a neuron.
    edge : tuple
        Edge contained in "graph".

    Returns
    -------
    tuple
        Indices of both nodes contained in "edge".

    """
    return get_idx(graph, edge[0]), get_idx(graph, edge[1])


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
