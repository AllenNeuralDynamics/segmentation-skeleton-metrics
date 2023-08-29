# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import networkx as nx
import numpy as np

from segmentation_skeleton_metrics import nx_utils, utils


def make_entries(graph, edge_list, anisotropy):
    """
    Makes a list of entries to be written in an swc file.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that edges in "edge_list" belong to.
    edge_list : list[tuple[int]]
        List of edges to be written to an swc file.
    anisotropy : list[float]
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files.

    Returns
    -------
    list[str]
        List of swc file entries to be written.

    """
    reindex = dict()
    for i, j in edge_list:
        if len(reindex) < 1:
            entry, reindex = make_entry(graph, i, -1, reindex, anisotropy)
            entry_list = [entry]
        entry, reindex = make_entry(graph, j, reindex[i], reindex, anisotropy)
        entry_list.append(entry)
    return entry_list


def make_entry(graph, i, parent, reindex, anisotropy):
    """
    Makes an entry to be written in an swc file.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that "i" and "parent" belong to.
    i : int
        Node that entry corresponds to.
    parent : int
         Parent of node "i".
    anisotropy : list[float]
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files.

    """
    reindex[i] = len(reindex) + 1
    x, y, z = tuple(map(str, node_to_world(graph, i, anisotropy)))
    return [x, y, z, 8, parent], reindex


def node_to_world(graph, i, anisotropy):
    """
    Converts "xyz" from image coordinates to real-world coordinates.

    Parameters
    ----------
    i : int
        Node to be querried for (x, y, z) coordinates which are then converted
        to read-world coordinates.
    anisotropy : list[float]
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files.

    Returns
    -------
    list[float]
        Transformed coordinates.

    """
    return to_world(nx_utils.get_xyz(graph, i), anisotropy)


def to_world(xyz, anisotropy):
    """
    Converts "xyz" from image coordinates to real-world coordinates.

    Parameters
    ----------
    xyz : tuple or list
        Coordinates to be transformed.
    anisotropy : list[float]
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files.

    Returns
    -------
    list[float]
        Transformed coordinates.

    """
    return [xyz[i] * anisotropy[i] for i in range(3)]


def write_swc(path, entry_list, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path : str
        Path on local machine that swc file will be written to.
    entry_list : list[list]
        List of entries that will be written to an swc file.
    color : str, optional
        Color of nodes. The default is None.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        if color is not None:
            f.write("# COLOR" + color)
        else:
            f.write("# id, type, z, y, x, r, pid")
        f.write("\n")
        for i, entry in enumerate(entry_list):
            f.write(str(i + 1) + " " + str(0) + " ")
            for x in entry:
                f.write(str(x) + " ")
            f.write("\n")


def dir_to_graphs(swc_dir, anisotropy=[1.0, 1.0, 1.0]):
    """
    Converts directory of swc files to a list of graphs.

    Parameters
    ----------
    swc_dir : str
        Path to directory containing swc files.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files.

    Returns
    -------
    list_of_graphs : list[networkx.Graph]
        List of graphs where each graph represents a neuron.

    """
    list_of_graphs = []
    for graph_id, f in enumerate(utils.listdir(swc_dir, ext=".swc")):
        path = os.path.join(swc_dir, f)
        graph = nx.Graph(file_name=f, graph_id=graph_id)
        graph = file_to_graph(path, graph, anisotropy=anisotropy)
        list_of_graphs.append(graph)
    return list_of_graphs


def file_to_graph(path, graph, anisotropy=[1.0, 1.0, 1.0]):
    """
    Reads an swc file and constructs an undirected graph from it.

    Parameters
    ----------
    path : str
        Path to swc file to be read.
    graph : networkx.Graph
        Graph that info from swc file will be stored as a graph.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files.

    Returns
    -------
    networkx.Graph
        Graph constructed from an swc file.

    """
    with open(path, "r") as f:
        offset = [0, 0, 0]
        for line in f.readlines():
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = read_xyz(parts[2:5])
            if not line.startswith("#") and len(line) > 0:
                parts = line.split()
                child = int(parts[0])
                parent = int(parts[-1])
                xyz = read_xyz(
                    parts[2:5], anisotropy=anisotropy, offset=offset
                )
                graph.add_node(child, xyz=xyz, radius=parts[-2])
                if parent != -1:
                    graph.add_edge(parent, child)
    return graph


def read_xyz(xyz, anisotropy=[1.0, 1.0, 1.0], offset=[0, 0, 0]):
    """
    Reads the (x,y,z) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    xyz : str
        (x,y,z) coordinates.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors applied to "xyz".
    offset : list[int], optional
        Offset of (x, y, z) coordinates in swc file.

    Returns
    -------
    tuple
        The (x,y,z) coordinates of an entry from an swc file in real-world
        coordinates.

    """
    xyz = [
        int(np.round(float(xyz[i]) / anisotropy[i] + offset[i]))
        for i in range(3)
    ]
    return tuple(xyz)