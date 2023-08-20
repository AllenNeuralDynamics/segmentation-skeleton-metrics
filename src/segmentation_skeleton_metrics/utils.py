# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
import json
import numpy as np
import shutil
import tensorstore as ts
import zarr
from tifffile import imread


# -- os utils ---
def listdir(path, ext=None):
    if ext is None:
        return [f for f in os.listdir(path)]
    else:
        return [f for f in os.listdir(path) if ext in f]


def mkdir(path):
    """
    Makes a directory at "path" if it does not already exist.

    Parameters
    ----------
    path : str
        Path to directory.

    Returns
    -------
    None.

    """
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# --- data structure utils ---
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
    if edge in set_of_edges or (edge[1], edge[0]) in set_of_edges:
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


# --- io utils ---
def read_tensorstore(path):
    """
    Uploads segmentation mask stored as a directory of shard files.

    Parameters
    ----------
    path : str
        Path to directory containing shard files.

    Returns
    -------
    sparse_volume : dict
        Sparse image volume.

    """
    dataset_ts = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": path},
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


def read_n5(path):
    """
    Uploads n5 file.

    Parameters
    ----------
    path : str
        Path to n5.

    Returns
    -------
    np.array
        Image volume.
    """
    return zarr.open(zarr.N5FSStore(path), "r").volume


def read_tif(path):
    """
    Uploads tif file.

    Parameters
    ----------
    path : str
        Path to tif file.

    Returns
    -------
    np.array
        Image volume.
    """
    return imread(path)


def write_txt(path, contents):
    with open(path, "w") as file:
        for line in contents:
            file.write(line + "\n")


def write_json(path, contents):
    with open(path, "w") as f:
        json.dump(contents, f)
