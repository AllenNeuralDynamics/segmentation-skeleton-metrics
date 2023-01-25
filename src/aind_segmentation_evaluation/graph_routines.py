# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from random import sample

import networkx as nx
import numpy as np
from scipy.ndimage.morphology import grey_dilation
from skimage.morphology import skeletonize_3d

from aind_segmentation_evaluation.utils import get_idx, get_xyz


# Conversion Routines
def graph_to_volume(list_of_graphs, shape):
    """
    Converts "list_of_graphs" to a sparse image volume.

    Parameters
    ----------
    list_of_graphs : list[networkx.Graph]
        List of graphs where each graph represents a neuron.
    shape : tuple
        Dimensions of "volume" in the order of (x,y,z).

    Returns
    -------
    dict
        Sparse image volume.

    """
    num_dilations = 3
    volume = graph_to_skeleton(list_of_graphs, shape)
    for _ in range(num_dilations):
        volume = grey_dilation(volume, mode="constant", size=(3, 3, 3))
    return volume_to_dict(volume)


def graph_to_skeleton(list_of_graphs, shape):
    """
    Converts "list_of_graphs" to an image volume by populating an array with
    the (x,y,z) coordinates of each node.

    Parameters
    ----------
    list_of_graphs : list[networkx.Graph]
        List of graphs where each graph represents a neuron.
    shape : tuple
        Dimensions of "volume" in the order of (x,y,z).

    Returns
    -------
    volume : numpy.array
        Image volume.

    """
    volume = np.zeros(shape, dtype=np.uint32)
    for i, graph in enumerate(list_of_graphs):
        volume = embed_graph(graph, volume, i + 1)
    return volume


def graph_to_swc(graph, path):
    """
    Converts graph to an swc file.

    Parameters
    ----------
    graph : networkx.Graph
        Graph which represents a neuron.
    path : str
        Path that swc file will be written to.

    Returns
    -------
    None.

    """
    root = sample(list(graph.nodes), 1)[0]
    swc = []
    queue = [(-1, root)]
    visited = set()
    reindex = dict()
    while len(queue) > 0:
        parent, child = queue.pop(0)
        swc.append(get_swc_entry(get_xyz(graph, child), 2, parent))
        visited.add(child)
        reindex[child] = len(swc)
        for nb in list(graph.neighbors(child)):
            if nb not in visited:
                queue.append((reindex[child], nb))
    write_swc(path, swc)


def skeleton_to_graph(skel):
    """
    Converts skeleton of a neuron to a graph.

    Parameters
    ----------
    skel : numpy.array
        Image volume of the skeleton of a neuron.

    Returns
    -------
    graph : networkx.Graph
        Graphical representation of "skel".

    """
    i, j, k = np.nonzero(skel)
    search_space = set([(i[n], j[n], k[n]) for n in range(len(i))])
    queue = [(-1, search_space.pop())]
    visited = []
    graph = nx.Graph()
    while len(queue) > 0:
        # Visit node
        parent_id, child_idx = queue.pop(0)
        child_id = graph.number_of_nodes() + 1
        graph.add_node(child_id, idx=child_idx, xyz=child_idx)
        if parent_id != -1:
            graph.add_edge(parent_id, child_id)
        visited.append(child_idx)

        # Populate queue
        for edge in get_bfs_nbs():
            nb_idx = get_nb(child_idx, edge)
            if nb_idx in search_space:
                search_space.remove(nb_idx)
                queue.append((child_id, nb_idx))
    return graph


def swc_to_graph(swc_dir, shape):
    """
    Converts directory of swc files to a list of graphs.

    Parameters
    ----------
    swc_dir : str
        Path to directory containing swc files.
    shape : tuple
        Dimensions of image volume in the order of (x,y,z).

    Returns
    -------
    list_of_graphs : list[networkx.Graph]
        List of graphs where each graph represents a neuron.

    """
    list_of_graphs = []
    for graph_id, f in enumerate(
        [f for f in os.listdir(swc_dir) if "swc" in f]
    ):
        graph = nx.Graph(file_name=f, graph_id=graph_id)
        with open(os.path.join(swc_dir, f), "r") as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                parts = line.split()
                child = int(parts[0])
                parent = int(parts[-1])
                xyz = read_xyz(parts[2:5])
                idx = read_idx(xyz, shape)
                graph.add_node(child, xyz=xyz, idx=idx)
                if parent != -1:
                    graph.add_edge(parent, child)
        list_of_graphs.append(graph)
    return list_of_graphs


def volume_to_dict(volume):
    """
    Converts an image volume to a dictionary (i.e. sparsifies).

    Parameters
    ----------
    volume : numpy.array
        Image volume.

    Returns
    -------
    d : dict
        Sparse image volume.

    """
    d = dict()
    i, j, k = np.nonzero(volume)
    for n in range(len(i)):
        idx = (i[n], j[n], k[n])
        d[idx] = volume[idx]
    return d


def volume_to_graph(volume):
    """
    Converts image to a list of graphs, where each label in the image
    corresponds to a distinct graph.

    Parameters
    ----------
    volume : numpy.array
        Image volume.

    Returns
    -------
    list_of_graphs : list[networkx.Graph]
        List of graphs where each graph corresponds to a neuron.

    """
    list_of_graphs = []
    binary_skeleton = skeletonize_3d(volume > 0).astype(int)
    skeleton = volume * binary_skeleton
    for i in [i for i in np.unique(skeleton) if i != 0]:
        mask_i = (skeleton == i).astype(int)
        graph_i = skeleton_to_graph(mask_i)
        list_of_graphs.append(graph_i)
    return list_of_graphs


def embed_graph(graph, volume, val, root=1):
    """
    Populates an array at the index of each node. Each entry is
    set to the value "val".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a neuron.
    volume : numpy.array
        Image volume.
    val : int
        Value that each populated entry is set to.
    root : int, optional
        Root node of "graph".
        The default is 1.

    Returns
    -------
    volume : numpy.array
        Image volume.

    """
    volume[get_idx(graph, root)] = val
    for (i, j) in nx.bfs_edges(graph, root):
        volume[get_idx(graph, j)] = val
    return volume


def get_bfs_nbs(nbhd=26):
    """
    Gets list of tuples such that each is the translation vector between the
    origin (i.e. (0,0,0)) and one of its of neighbors.

    Parameters
    ----------
    nbhd : int, optional
        Connectivity of 3D neighborhood (e.g. 6, 18, 26).
        The default is 26.

    Returns
    -------
    nbs : list[tuple]
        list of translation vectors of neighbors of the origin.

    """
    nbs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    if nbhd >= 18:
        l1 = [(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0)]
        l2 = [(1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1)]
        l3 = [(0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)]
        nbs = nbs + l1 + l2 + l3
    if nbhd == 26:
        l1 = [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]
        l2 = [(-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
        l3 = [(1, 1, 1), (-1, -1, -1)]
        nbs = nbs + l1 + l2 + l3
    return nbs


def get_nb(xyz, vec):
    """
    Gets neighbor of node with coordinates "xyz" by adding "vec" to "xyz".

    Parameters
    ----------
    xyz : tuple
        (x,y,z) coordinates of some node.
    vec : tuple
        Vector.

    Returns
    -------
    TYPE
        Neighbor of node with coordinates "xyz".

    """
    return tuple([int(sum(i)) for i in zip(xyz, vec)])


# SWC read/write routines
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


def get_swc_entry(xyz, radius, parent):
    """
    Gets text (i.e. "entry") that will be written to an swc file.

    Parameters
    ----------
    xyz : tuple
        (x,y,z) coordinates of node corresponding to this entry.
    radius : int
        Size of node written to swc file.
    parent : int
        Parent of node corresponding to this entry.

    Returns
    -------
    entry : str
        Text (i.e. "entry") that will be written to an swc file.

    """
    entry = [val * get_scaling_factor() for val in xyz]
    entry.reverse()
    entry.extend([radius, int(parent)])
    return entry


def read_xyz(zyx):
    """
    Reads the (z,y,x) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    zyx : str
        (z,y,x) coordinates.

    Returns
    -------
    zyx_scaled : tuple
        The (x,y,z) coordinates from an swc file.

    """
    zyx_scaled = [float(val) / get_scaling_factor() for val in zyx]
    zyx_scaled.reverse()
    return zyx_scaled


def read_idx(xyz, shape):
    """
    Converts (x,y,z) coordinates to indices (row, column, depth).

    Parameters
    ----------
    xyz : tuple
        (x,y,z) coordinates.
    shape : tuple
        (length, width, height) of image volume.

    Returns
    -------
    list
        Indices computed from "xyz".

    """
    return [intergize(val, shape[i] - 1) for i, val in enumerate(xyz)]


def intergize(val, dim):
    """
    Converts "val" to an integer and ensures that it is contained within
    dimension (i.e. "dim") of image volume.

    Parameters
    ----------
    val : int
        Value at voxel.
    dim : int
        Dimension of image volume.

    Returns
    -------
    int
        Index computed from val.

    """
    idx = min(max(np.round(val), 0), dim)
    return int(idx)


def get_scaling_factor():
    """
    Gets scaling factor used in Janelia Workstation

    Returns
    -------
    float
        Scaling factor.

    """
    return 1.1010572351571979
