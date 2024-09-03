# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
import shutil
from io import BytesIO
from zipfile import ZipFile

import networkx as nx
import tensorstore as ts

ANISOTROPY = [0.748, 0.748, 1.0]
MIN_CNT = 20
SUPPORTED_DRIVERS = ["neuroglancer_precomputed", "n5", "zarr"]


# -- os utils ---
def mkdir(path, delete=False):
    """
    Creates a directory at "path".

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. The default is False.

    Returns
    -------
    None

    """
    if delete:
        rmdir(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    """
    Removes directory and all subdirectories at "path".

    Parameters
    ----------
    path : str
        Path to directory and subdirectories to be deleted if they exist.

    Returns
    -------
    None

    """
    if os.path.exists(path):
        shutil.rmtree(path)


def list_dir(directory, extension=None):
    """
    Lists all filenames in "directory". If "extension" is provided, then only
    filenames ending with "extension" are returned.

    Parameters
    ----------
    directory : str
        Path to directory to be searched.

    extension : str, optional
       File type of interest. The default is None.

    Returns
    -------
    list[str]
        Filenames in the directory "directory".

    """
    if extension is None:
        return [f for f in os.listdir(directory)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(extension)]


def list_paths(directory, extension=None):
    """
    Lists all paths of files in the directory "directory". If "extension" is
    provided, then only paths of files ending with "extension" are returned.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    extension : str, optional
        File type of interest. The default is None.

    Returns
    -------
    list[str]
        Paths of files in the directory "directory".

    """
    paths = []
    for f in list_dir(directory, extension=extension):
        paths.append(os.path.join(directory, f))
    return paths


def get_id(path):
    """
    Gets segment id of the swc file at "path".

    Parameters
    ----------
    path : str
        Path to an swc file

    Return
    ------
    int or str
        Segment id of swc file.

    """
    filename = os.path.basename(path).split(".")[0]
    return int(filename) if filename.isdigit() else filename


# --- io utils ---
def open_tensorstore(path, driver):
    """
    Opens the tensorstore array stored at "path".

    Parameters
    ----------
    path : str
        Path to directory containing shard files.
    driver : str
        Storage driver needed to read data at "path".

    Returns
    -------
    dict
        Sparse arr.

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


def read_zip(zip_file, path):
    """
    Reads the content of an swc file from a zip file.

    Parameters
    ----------
    zip_file : ZipFile
        Zip containing text file to be read.

    Returns
    -------
    str
        Contents of a txt file.

    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


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
        Contents of a txt file.

    """
    with open(path, "r") as f:
        return f.readlines()


def list_gcs_filenames(bucket, cloud_path, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    Parameters
    ----------
    bucket : google.cloud.client
        Client used to read from a GCS bucket.
    cloud_path : str
        ...
    extension : str
        File type of interest. The default is None.

    Returns
    -------
    list[str]
        Filenames in directory specified by "cloud_path" that end with
        "extension".

    """
    blobs = bucket.list_blobs(prefix=cloud_path)
    return [blob.name for blob in blobs if blob.name.endswith(extension)]


def list_files_in_zip(zip_content):
    """
    Lists all filenames in a zip.

    Parameters
    ----------
    zip_content : ...
        ...

    Returns
    -------
    list
        Filenames in a zip.

    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


# -- dict utils --
def filter_dict(my_dict):
    """
    Filters a dictionary by removing entries with values that do not meet a
    minimum count threshold.

    Parameters
    ----------
    my_dict : dict
        A dictionary where each key is a set of items.

    Returns
    -------
    dict
        A new dictionary containing only the entries from "my_dict" where the
        length of the value is greater than "MIN_CNT".

    """
    return {k: v for k, v in my_dict.items() if len(v) > MIN_CNT}


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
    bool
        Indication of whether "edge" is contained in "edge_list".

    """
    if edge in edge_list or (edge[1], edge[0]) in edge_list:
        return True
    else:
        return False


def delete_keys(my_dict, keys):
    """
    Deletes a set of keys from a dictionary.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be updated.
    keys : container
        Keys to be deleted

    Returns
    -------
    dict
        Updated dictionary.

    """
    for key in keys:
        del my_dict[key]
    return my_dict


# -- build label graph --
def init_label_map(connections_path, labels):
    label_to_class = {0: 0}
    class_to_labels = {0: [0]}
    labels_graph = build_labels_graph(connections_path, labels)
    for i, component in enumerate(nx.connected_components(labels_graph)):
        i += 1
        class_to_labels[i] = set()
        for label in component:
            label_to_class[label] = i
            class_to_labels[i].add(label)
    return label_to_class, class_to_labels


def build_labels_graph(connections_path, labels):
    """
    Builds a graph from a list of labels and connection data. The nodes are
    initialized from the provided list of labels, then edges are added between
    nodes based on a list of connections specified in a file.

    Parameters
    ----------
    connections_path : str
        The file path to a text file containing connections. Each line should
        represent a connection between two swc ids.
    labels : iterable
        An iterable containing the IDs of the nodes to be added to the graph.

    Returns
    -------
    networkx.Graph
        Graph with nodes that represent labels and edges are based on the
        connections read from the "connections_path".

    """
    # Initializations
    labels_graph = nx.Graph()
    labels_graph.add_nodes_from(labels)
    n_components = nx.number_connected_components(labels_graph)

    # Main
    print("Building Label Graph...")
    print("# connected components - before adding edges:", n_components)
    for line in read_txt(connections_path):
        ids = line.split(",")
        id_1 = int(ids[0])
        id_2 = int(ids[1])
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
        Unit of "t". The default is "seconds".

    Returns
    -------
    float
        Converted runtime.
    str
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


def progress_bar(current, total, bar_length=50):
    """
    Reports the progress of completing some process.

    Parameters
    ----------
    current : int
        Current iteration of process.
    total : int
        Total number of iterations to be completed.
    bar_length : int, optional
        Length of progress bar. The default is 50.

    Returns
    -------
    None

    """
    progress = int(current / total * bar_length)
    bar = (
        f"[{'=' * progress}{' ' * (bar_length - progress)}] {current}/{total}"
    )
    print(f"\r{bar}", end="", flush=True)


# -- miscellaneous --
def to_world(voxel):
    """
    Converts coordinates from voxels to world.

    Parameters
    ----------
    voxel : numpy.ndarray
        Coordinate to be converted.

    Returns
    -------
    tuple
        Converted coordinates.

    """
    return tuple([voxel[i] * ANISOTROPY[i] for i in range(3)])
