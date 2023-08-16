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

from aind_segmentation_evaluation import graph_utils as gutils
from aind_segmentation_evaluation import swc_utils, utils


# Conversion Routines
def to_world(idx, anisotropy):
    """ "
    Converts "idx" to real-world coordinates.

    Parameters
    ----------
        idx : list[idx]
            Image indexes to be converted.
        anisotropy : list[float]
            Scaling factor that is applied to permuted "idx".
        shift : list[float]
            Shift that is applied to "idx".

    Returns
    -------
    list
        The result of applying this series of transformations to "idx".

    """
    return [idx[i] / anisotropy[i] for i in range(3)]


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
    numpy.array
        Image volume.

    """
    num_dilations = 3
    volume = graph_to_skeleton(list_of_graphs, shape)
    for _ in range(num_dilations):
        volume = grey_dilation(volume, mode="constant", cval=0, size=(3, 3, 3))
    return volume


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


def graphs_to_swc(
    list_of_graphs, root_dir, anisotropy=[1, 1, 1], shift=[0, 0, 0]
):
    """
    Converts list of graphs to a directory of swc files.

    Parameters
    ----------
    list_of_graphs : list[networkx.graph]
        List of graphs to be written to swc files. Each graph is written as
        its own swc file.
    root_dir : path
        Directory where swc files will be saved.
    anisotropy : list[int], optional
        Image to real-world coordinates scaling factors for (x, y, z) which is
        applied to swc files.
    shift : list[float], optional
        Shift that is applied to "idx".

    Returns
    -------
    None

    """
    mkdir(root_dir)
    cnt = 1
    for i, graph in enumerate(list_of_graphs):
        path = os.path.join(root_dir, str(i) + ".swc")
        graph_to_swc(graph, path, anisotropy=anisotropy, shift=shift)


def graph_to_swc(graph, path, scale=[1, 1, 1], shift=[0, 0, 0]):
    """
    Converts graph to an swc file.

    Parameters
    ----------
    graph : networkx.Graph
        Graph which represents a neuron.
    path : str
        Path that swc file will be written to.
    permute : list[int], optional
        Permutation from image to real-world coordinates.
    scale : list[int], optional
        Scaling factor from image to real-world coordinates.
    shift : list[float], optional
        Shift that is applied to "idx".

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
        try:
            radius = graph.nodes[child]["radius"]
        except:
            radius = 1
            print("Node missing radius")

        swc.append(
            swcr.make_entry(
                get_xyz(graph, child),
                radius,
                parent,
                permute=permute,
                scale=scale,
                shift=shift,
            )
        )
        visited.add(child)
        reindex[child] = len(swc)
        for nb in list(graph.neighbors(child)):
            if nb not in visited:
                queue.append((reindex[child], nb))
    swcr.write_swc(path, swc)


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
        graph.add_node(child_id, idx=child_idx, xyz=child_idx, radius=1)
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


def volume_to_graph(volume, min_branch_length=10, prune=True):
    """
    Converts image to a list of graphs, where each label in the image
    corresponds to a distinct graph.

    Parameters
    ----------
    volume : numpy.array
        Image volume.
    min_branch_length : int, optional
        Minimum length of branch that is not pruned.
    prune : bool, optional
        Indicates whether to prune spurious branches in graph.
        The default is True.

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
        if prune:
            graph_i = prune_spurious_paths(
                graph_i, min_branch_length=min_branch_length
            )
        list_of_graphs.append(graph_i)
    return list_of_graphs


def embed_graph(graph, volume, val):
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

    Returns
    -------
    volume : numpy.array
        Image volume.

    """
    for i in graph.nodes:
        volume[get_idx(graph, i)] = val
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
    tuple
        Neighbor of node with coordinates "xyz".

    """
    return tuple([int(sum(i)) for i in zip(xyz, vec)])
