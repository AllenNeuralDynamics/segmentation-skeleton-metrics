# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import networkx as nx
import numpy as np

from segmentation_skeleton_metrics import utils


def read(path, cloud_read=False):
    return read_from_cloud(path) if cloud_read else read_from_local(path)


def read_from_cloud(path):
    pass


def read_from_local(path):
    """
    Reads swc file stored at "path" on local machine.

    Parameters
    ----------
    Path : str
        Path to swc file to be read.

    Returns
    -------
    list
        List such that each entry is a line from the swc file.

    """
    with open(path, "r") as file:
        return file.readlines()


def get_xyz_coords(path, anisotropy=[1.0, 1.0, 1.0]):
    """
    Gets the xyz coords from the swc file at "path".

    Parameters
    ----------
    path : str
        Path to swc file to be parsed.
    anisotropy : list[float], optional
        Scaling factors applied to xyz coordinates to account for anisotropy
        of the microscope. The default is [1.0, 1.0, 1.0].

    Returns
    -------
    numpy.ndarray
        xyz coords from an swc file.

    """
    xyz_list = []
    with open(path, "r") as f:
        offset = [0, 0, 0]
        for line in f.readlines():
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = read_xyz(parts[2:5])
            if not line.startswith("#"):
                parts = line.split()
                xyz = read_xyz(
                    parts[2:5], anisotropy=anisotropy, offset=offset
                )
                xyz_list.append(xyz)
    return np.array(xyz_list)


def read_xyz(xyz, anisotropy=[1.0, 1.0, 1.0], offset=[0, 0, 0]):
    """
    Reads the (x,y,z) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    xyz : str
        (x,y,z) coordinates.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors applied to "xyz". The
        default is [1.0, 1.0, 1.0].
    offset : list[int], optional
        Offset of (x, y, z) coordinates in swc file. The default is [0, 0, 0].

    Returns
    -------
    tuple
        The (x,y,z) coordinates of an entry from an swc file in real-world
        coordinates.

    """
    xyz = [float(xyz[i]) + offset[i] for i in range(3)]
    return np.array([xyz[i] / anisotropy[i] for i in range(3)], dtype=int)


def save(path, xyz_1, xyz_2, color=None):
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
        # Preamble
        if color is not None:
            f.write("# COLOR " + color)
        else:
            f.write("# id, type, z, y, x, r, pid")
        f.write("\n")

        # Entries
        f.write(make_entry(1, -1, xyz_1))
        f.write("\n")
        f.write(make_entry(2, 1, xyz_2))


def make_entry(node_id, parent_id, xyz):
    """
    Makes an entry to be written in an swc file.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that "node_id" and "parent_id" belong to.
    node_id : int
        Node that entry corresponds to.
    parent_id : int
         Parent of node "node_id".
    xyz : ...
        xyz coordinate of "node_id".

    Returns
    -------
    entry : str
        Entry to be written in an swc file.

    """
    x, y, z = tuple(xyz)
    entry = f"{node_id} 2 {x} {y} {z} 8 {parent_id}"
    return entry


def to_graph(path, anisotropy=[1.0, 1.0, 1.0]):
    """
    Reads an swc file and builds an undirected graph from it.

    Parameters
    ----------
    path : str
        Path to swc file to be read.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files. The default is [1.0, 1.0, 1.0].

    Returns
    -------
    networkx.Graph
        Graph built from an swc file.

    """
    graph = nx.Graph(swc_id=utils.get_swc_id(path))
    offset = [0, 0, 0]    
    for line in read(path):
        if line.startswith("# OFFSET"):
            parts = line.split()
            offset = read_xyz(parts[2:5])
        if not line.startswith("#"):
            parts = line.split()
            child = int(parts[0])
            parent = int(parts[-1])
            xyz = read_xyz(parts[2:5], anisotropy=anisotropy, offset=offset)
            graph.add_node(child, xyz=xyz)
            if parent != -1:
                graph.add_edge(parent, child)
    return graph
