# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import numpy as np
from aind_segmentation_evaluation.conversions import to_world


def make_entry(
    xyz,
    radius,
    parent,
    permute=[0, 1, 2],
    scale=[1.0, 1.0, 1.0],
    shift=[0, 0, 0],
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
    permute : list[int], optional
        Permutation that is applied to "idx". The default is None.
    scale : list[float], optional
        Scaling factor that is applied to permuted "idx". The default is None.
    shift : list[float], optional
        Shift that is applied to "idx". The default is None.

    Returns
    -------
    entry : str
        Text (i.e. "entry") that will be written to an swc file.

    """
    entry = to_world(xyz, permute, scale, shift)
    entry.extend([2, int(parent)])
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


def read_xyz(xyz, scale, permute):
    """
    Reads the (z,y,x) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    zyx : str
        (z,y,x) coordinates.
    scale : list[float]
        Image to real-world coordinates scaling factors for [x, y, z].
    permute : list[int]
        Permutation that is applied to "idx".

    Returns
    -------
    zyx_scaled : tuple
        The (x,y,z) coordinates from an swc file.

    """
    xyz = list(map(float, xyz))
    xyz = [xyz[i] for i in permute]
    xyz = [xyz[i] / scale[i] for i in range(3)]
    return xyz


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
