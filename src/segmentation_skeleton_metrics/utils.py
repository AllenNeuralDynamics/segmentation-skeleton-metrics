# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from io import BytesIO
from time import time
from zipfile import ZipFile

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import distance

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
    """
    Lists all paths within "directory" with extension "ext" if provided.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    ext : str, optional
        If provided, only paths of files with the extension "ext" are
        returned. The default is None.

    Returns
    -------
    list[str]
        List of all paths within "directory" with extension "ext" if provided.

    """
    paths = []
    for f in listdir(directory, ext=ext):
        paths.append(os.path.join(directory, f))
    return paths


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
    assert driver in SUPPORTED_DRIVERS, "Driver is not supported!"
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
        Path to directory containing shardsS.

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


def read_from_gcs_zip(zip_file, path):
    """
    Reads the content of an swc file from a zip file in a GCS bucket.

    """
    with zip_file.open(path) as text_file:
        return text_file.read().decode("utf-8").splitlines()


def read_txt(path):
    """
    Reads txt file stored at "path".

    Parameters
    ----------
    path : str
        Path where txt file is stored.

    Returns
    -------
    str
        Contents of txt file.

    """
    with open(path, "r") as f:
        return f.readlines()


def list_gcs_filenames(bucket, cloud_path, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    """
    blobs = bucket.list_blobs(prefix=cloud_path)
    return [blob.name for blob in blobs if extension in blob.name]


def list_files_in_gcs_zip(zip_content):
    """
    Lists all files in a zip file stored in a GCS bucket.

    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


# -- miscellaneous --
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


def dist(v_1, v_2):
    """
    Computes distance between "v_1" and "v_2".

    Parameters
    ----------
    v_1 : np.ndarray
        Vector.
    v_2 : np.ndarray
        Vector.

    Returns
    -------
    float
        Distance between "v_1" and "v_2".

    """
    return distance.euclidean(v_1, v_2)


def get_midpoint(xyz_1, xyz_2):
    """
    Computes the midpoint between "xyz_1" and "xyz_2".

    Parameters
    ----------
    xyz_1 : numpy.ndarray
        xyz coordinate.
    xyz_2 : numpy.ndarray
        xyz coordinate.

    Returns
    -------
    numpy.ndarray
        Midpoint between "xyz_1" and "xyz_2".

    """
    return np.array([np.mean([xyz_1[i], xyz_2[i]]) for i in range(3)])


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


def above_threshold(my_dict):
    # Find keys to delete
    delete_keys = list()
    for key, value in my_dict.items():
        if len(value) < 16:
            delete_keys.append(key)

    # Delete keys
    while len(delete_keys) > 0:
        key = delete_keys.pop()
        del my_dict[key]

    return my_dict


def resolve(multi_hits, dists, xyz_to_id_node):
    """
    Resolves discrepancy when xyz coordinates project onto ground truth graphs
    that contain nodes with same xyz coordinates.

    Parameters
    ----------
    multi_hits : set
        xyz coordinates that are common across multiple graphs.
    dists : dict
        Dictionary containing graph ids that predicted swc file has
        intersected.
    xyz_to_id_node : dict
        Dictionary where the keys are xyz coordinates that correspond to nodes
        in ground truth graphs and values are the graph ids and nodes.

    Return
    ------
    dists : dict
        Updated dictionary with discrepancies resolved.

    """
    for hat_xyz in multi_hits:
        keys = list(xyz_to_id_node[hat_xyz].keys())
        id = find_best(dists, keys)
        if id:
            node = xyz_to_id_node[hat_xyz][id]
            dists = append_dict_value(dists, id, node)
    return dists


def append_dict_value(my_dict, key, value):
    """
    Appends "value" to the list stored at "key".

    Parameters
    ----------
    my_dict : dict
        Dictionary to be queried.
    key : hashable data type
        Key to be query.
    value : list item type
        Value to append to list stored at "key".

    Returns
    -------
    my_dict : dict
        Updated dictionary.

    """
    if key in my_dict.keys():
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
    return my_dict


def find_best(my_dict, keys):
    best_key = None
    best_vote_cnt = 0
    if len(my_dict) > 0:
        for key in keys:
            vote_cnt = len(my_dict[key]) if key in my_dict.keys() else 0
            if vote_cnt > best_vote_cnt:
                best_key = key
                best_vote_cnt = vote_cnt
    return best_key


def get_id(path):
    """
    Gets segment id of the swc file at "path".

    Parameters
    ----------
    path : str
        Path to an swc file

    Return
    ------
    Segment id of swc file.

    """
    filename = os.path.basename(path)
    return filename.split(".")[0]


def init_label_map(connections_path, labels):
    label_to_class = {0: 0}
    labels_graph = build_labels_graph(connections_path, labels)
    for i, component in enumerate(nx.connected_components(labels_graph)):
        i += 1
        for label in component:
            label_to_class[label] = i
    return label_to_class


def build_labels_graph(connections_path, labels):
    # Initializations
    labels_graph = nx.Graph()
    labels_graph.add_nodes_from(labels)
    n_components = nx.number_connected_components(labels_graph)

    # Main
    print("Building label graph...")
    print("# connected components - before adding edges:", n_components)
    for line in read_txt(connections_path):
        ids = line.split(",")
        id_1 = int(ids[0])
        id_2 = int(ids[1])
        print(id_1, id_2)
        stop
        assert id_1 in labels_graph.nodes
        assert id_2 in labels_graph.nodes
        labels_graph.add_edge(id_1, id_2)
    n_components = nx.number_connected_components(labels_graph)
    print("# connected components - after adding edges:", n_components)
    print("")
    return labels_graph


# -- runtime --
def time_writer(t, unit="seconds"):
    """
    Converts runtime "t" to its proper unit.

    Parameters
    ----------
    t : float
        Runtime to be converted.
    unit : str, optional
        Unit of "t".

    Returns
    -------
    t : float
        Converted runtime.
    unit : str
        Unit of "t"

    """
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit


def init_timers():
    """
    Initializes two timers.

    Parameters
    ----------
    None

    Returns
    -------
    time.time
        Timer.
    time.time
        Timer.

    """
    return time(), time()


# -- Utils --
def report_progress(current, total, chunk_size, cnt, t0, t1):
    eta = get_eta(current, total, chunk_size, t1)
    runtime = get_runtime(current, total, chunk_size, t0, t1)
    progress_bar2(current, total, eta=eta, runtime=runtime)
    return cnt + 1, time()


def get_eta(current, total, chunk_size, t0, return_str=True):
    chunk_runtime = time() - t0
    remaining = total - current
    eta = remaining * (chunk_runtime / chunk_size)
    t, unit = time_writer(eta)
    return f"{round(t, 4)} {unit}" if return_str else eta


def get_runtime(current, total, chunk_size, t0, t1):
    eta = get_eta(current, total, chunk_size, t1, return_str=False)
    total_runtime = time() - t0 + eta
    t, unit = time_writer(total_runtime)
    return f"{round(t, 4)} {unit}"


def progress_bar(current, total, bar_length=50):
    """
    Reports the progress of completing some process.

    Parameters
    ----------
    current : int
        Current iteration of process.
    total : int
        Total number of iterations to be completed
    bar_length : int, optional
        Length of progress bar

    Returns
    -------
    None

    """
    progress = int(current / total * bar_length)
    bar = (
        f"[{'=' * progress}{' ' * (bar_length - progress)}] {current}/{total}"
    )
    print(f"\r{bar}", end="", flush=True)

def progress_bar2(current, total, bar_length=50, eta=None, runtime=None):
    progress = int(current / total * bar_length)
    n_completed = f"Completed: {current}/{total}"
    bar = f"[{'=' * progress}{' ' * (bar_length - progress)}]"
    eta = f"Time Remaining: {eta}" if eta else ""
    runtime = f"Estimated Total Runtime: {runtime}" if runtime else ""
    print(f"\r{bar} {n_completed} | {eta} | {runtime}    ", end="", flush=True)
