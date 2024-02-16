# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import json
import os
import shutil

import tensorstore as ts
import zarr
from tifffile import imread

SUPPORTED_DRIVERS = ["neuroglancer_precomputed", "n5", "zarr"]


# -- os utils ---
def listdir(directory, ext=None):
    """
    Lists all files in "directory". If an extension "ext" is
    provided, then only files containing "ext" are returned.

    Parameters
    ----------
    directory : str
        Path to directory to be searched.

    ext : str, optional
       Extension of file type of interest. The default is None.

    Returns
    -------
    list
        List of all files in directory at "path" with extension "ext" if
        provided. Otherwise, list of all files in directory.

    """
    if ext is None:
        return [f for f in os.listdir(directory)]
    else:
        return [f for f in os.listdir(directory) if ext in f]


def list_paths(directory, ext=None):
    paths = []
    for f in listdir(directory, ext=ext):
        paths.append(os.path.join(directory, f))
    return paths
    

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
    """
    Removes a directory at "path" if it already exist.

    Parameters
    ----------
    path : str
        Path to directory.

    Returns
    -------
    None.

    """
    if os.path.exists(path):
        shutil.rmtree(path)


# --- data structure utils ---
def check_edge(edge_list, edge):
    """
    Checks if "edge" is in "edge_list".

    Parameters
    ----------
    edge_list : list or set
        List or set of edges.
    edge : tuple
        Edge.

    Returns
    -------
    bool : bool
        Indication of whether "edge" is contained in "edge_list".

    """
    if edge in edge_list or (edge[1], edge[0]) in edge_list:
        return True
    else:
        return False


# --- io utils ---
def open_tensorstore(path, driver):
    """
    Uploads segmentation mask stored as a directory of shard files.

    Parameters
    ----------
    path : str
        Path to directory containing shard files.
    driver : str
        Storage driver needed to read data at "path".

    Returns
    -------
    sparse_volume : dict
        Sparse image volume.

    """
    assert driver in SUPPORTED_DRIVERS, "Error! Driver is not supported!"
    arr = ts.open(
        {
            "driver": driver,
            "kvstore": {
                "driver": "gcs",
                "bucket": "allen-nd-goog",
                "path": path,
            },
            "context": {
                "cache_pool": {"total_bytes_limit": 1000000000},
                "cache_pool#remote": {"total_bytes_limit": 1000000000},
                "data_copy_concurrency": {"limit": 8},
            },
            "recheck_cached_data": "open",
        }
    ).result()
    if driver == "neuroglancer_precomputed":
        return arr[ts.d["channel"][0]]
    return arr


def read_tensorstore(path):
    """
    Reads neuroglancer_precomputed file at "path".

    Parameters
    ----------
    path : str
        Path to directory containing shard files.

    Returns
    -------
    ts.TensorStore
        Image volume.

    """
    dataset_ts = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": path},
        }
    ).result()
    return dataset_ts[ts.d["channel"][0]]


def read_tensorstore(arr, xyz, shape, from_center=True):
    chunk = get_chunk(arr, xyz, shape, from_center=from_center)
    return chunk.read().result()


def get_chunk(arr, xyz, shape, from_center=True):
    start, end = get_start_end(xyz, shape, from_center=from_center)
    return deepcopy(
        arr[start[0]: end[0], start[1]: end[1], start[2]: end[2]]
    )


def read_img(path, filetype):
    if filetype == "tensorstore":
        return read_tensorstore(path)
    elif filetype == "n5":
        return read_n5(path)
    else:
        return read_tif(path)


def read_n5(path):
    """
    Reads n5 file at "path".

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
    Reads tif file at "path".

    Parameters
    ----------
    path : str
        Path to tif.

    Returns
    -------
    np.array
        Image volume.
    """
    return imread(path)


def write_txt(path, contents):
    """
    Writes "contents" to a .txt file at "path".

    Parameters
    ----------
    path : str
        Path that .txt file is written to.
    contents : list[str]
        Contents to be written to .txt file.

    Returns
    -------
    None

    """
    with open(path, "w") as file:
        for line in contents:
            file.write(line + "\n")


def write_json(path, contents):
    """
    Writes "contents" to a .json file at "path".

    Parameters
    ----------
    path : str
        Path that .txt file is written to.
    contents : dict
        Contents to be written to .txt file.

    Returns
    -------
    None

    """
    with open(path, "w") as f:
        json.dump(contents, f)

        
# -- miscellaneous --
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


def time_writer(t, unit="seconds"):
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit
