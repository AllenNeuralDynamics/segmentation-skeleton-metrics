# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import networkx as nx
import numpy as np

import segmentation_skeleton_metrics.conversions as conv
from segmentation_skeleton_metrics import utils


def make_entry(
    xyz, radius, parent, anisotropy=[1.0, 1.0, 1.0], shift=[0, 0, 0]
):
    """
    Generates text (i.e. "entry") that will be written to an swc file.

    Parameters
    ----------
    xyz : tuple
        (x,y,z) coordinates of node corresponding to this entry.
    radius : int
        Size of node written to swc file.
    parent : int
        Parent of node corresponding to this entry.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors which are
        applied to "xyz".

    Returns
    -------
    entry : str
        Text (i.e. "entry") that will be written to an swc file.

    """
    entry = conv.to_world(xyz, anisotropy)
    entry.extend([radius, int(parent)])
    return entry


def write_swc(path_to_swc, list_of_entries, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path_to_swc : str
        Path that swc will be written to.
    list_of_entries : list[list[int]]
        List of entries that will be written to an swc file.
    color : str, optional
        Color of nodes. The default is None.

    Returns
    -------
    None.

    """
    with open(path_to_swc, "w") as f:
        if color is not None:
            f.write("# COLOR" + color)
        else:
            f.write("# id, type, z, y, x, r, pid")
        f.write("\n")
        for i, entry in enumerate(list_of_entries):
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
    shape : tuple
        Dimensions of image volume in the order of (x, y, z).
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors for (x, y, z) which is
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
    offset = [0.0, 0.0, 0.0]
    with open(path, "r") as f:
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


def read_xyz(xyz, anisotropy=[1.0, 1.0, 1.0], offset=[1.0, 1.0, 1.0]):
    """
    Reads the (x,y,z) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    xyz : str
        (x,y,z) coordinates.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors for [x, y, z] due to
        anistropy of the microscope.

    Returns
    -------
    list
        The (x,y,z) coordinates from an swc file.

    """
    xyz = [int(float(xyz[i]) * anisotropy[i] + offset[i]) for i in range(3)]
    return tuple(xyz)
