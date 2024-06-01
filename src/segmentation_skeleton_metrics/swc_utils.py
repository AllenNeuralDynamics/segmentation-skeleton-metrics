# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from zipfile import ZipFile

import networkx as nx
import numpy as np
from google.cloud import storage

from segmentation_skeleton_metrics import utils


def parse(swc_paths, min_size, anisotropy):
    """
    Reads swc files and extracts the xyz coordinates.

    Paramters
    ---------
    swc_paths : list or dict
        If swc files are on local machine, list of paths to swc files where
        each file corresponds to a neuron in the prediction. If swc files are
        on cloud, then dict with keys "bucket_name" and "path".
    min_size : int
        Threshold on the number of nodes contained in an swc file. Only swc
        files with more than "min_size" nodes are stored in "valid_labels".
    anisotropy : list[float]
        Image to World scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    dict
        Dictionary where each item is an swc_id and an array of the xyz
        coordinates read from cooresponding swc file.

    """
    if type(swc_paths) == list:
        return parse_local_paths(swc_paths, min_size, anisotropy)
    elif type(swc_paths) == dict:
        return parse_cloud_paths(swc_paths, min_size, anisotropy)
    else:
        return None


def parse_local_paths(swc_paths, min_size, anisotropy):
    """
    Reads swc files from local machine and extracts the xyz coordinates.

    Paramters
    ---------
    swc_paths : list or dict
        If swc files are on local machine, list of paths to swc files where
        each file corresponds to a neuron in the prediction. If swc files are
        on cloud, then dict with keys "bucket_name" and "path".
    min_size : int
        Threshold on the number of nodes contained in an swc file. Only swc
        files with more than "min_size" nodes are stored in "valid_labels".
    anisotropy : list[float]
        Image to World scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    valid_labels : dict
        Dictionary where each item is an swc_id and an array of the xyz
        coordinates read from cooresponding swc file.

    """
    valid_labels = dict()
    for path in swc_paths:
        contents = read_from_local(path)
        if len(contents) > min_size:
            id = int(utils.get_id(path))
            valid_labels[id] = get_coords(contents, anisotropy)
    return valid_labels


def parse_cloud_paths(cloud_dict, min_size, anisotropy):
    """
    Reads swc files from a GCS bucket and extracts the xyz coordinates.

    Parameters
    ----------
    cloud_dict : dict
        Dictionary where keys are "bucket_name" and "path".
    min_size : int
        Threshold on the number of nodes contained in an swc file. Only swc
        files with more than "min_size" nodes are stored in "valid_labels".
    anisotropy : list[float]
        Image to World scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    valid_labels : dict
        Dictionary where each item is an swc_id and an array of the xyz
        coordinates read from cooresponding swc file.

    """
    # Initializations
    bucket = storage.Client().bucket(cloud_dict["bucket_name"])
    zip_paths = utils.list_gcs_filenames(bucket, cloud_dict["path"], ".zip")
    print("Downloading predicted swc files from cloud...")
    print("# zip files:", len(zip_paths))

    # Assign processes
    chunk_size = int(len(zip_paths) * 0.02)
    cnt = 1
    t0, t1 = utils.init_timers()
    with ProcessPoolExecutor() as executor:
        processes = []
        for i, path in enumerate(zip_paths):
            zip_content = bucket.blob(path).download_as_bytes()
            processes.append(
                executor.submit(download, zip_content, anisotropy, min_size)
            )
            if i > cnt * chunk_size:
                cnt, t1 = utils.report_progress(
                    i, len(zip_paths), chunk_size, cnt, t0, t1
                )

    # Store results
    valid_labels = dict()
    for i, process in enumerate(as_completed(processes)):
        valid_labels.update(process.result())
    print("\n#Valid Labels:", len(valid_labels))
    return valid_labels


def download(zip_content, anisotropy, min_size):
    """
    Downloads the contents from each swc file contained in the zip file at
    "zip_path".

    Parameters
    ----------
    zip_content : ...
        Contents of a zip file.
    anisotropy : list[float]
        Image to World scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.
    min_size : int
        Threshold on the number of nodes contained in an swc file. Only swc
        files with more than "min_size" nodes are stored in "valid_labels".

    Returns
    -------
    valid_labels : dict
        Dictionary where each item is an swc_id and an array of the xyz
        coordinates read from cooresponding swc file.

    """
    with ZipFile(BytesIO(zip_content)) as zip_file:
        with ThreadPoolExecutor() as executor:
            # Assign threads
            paths = utils.list_files_in_gcs_zip(zip_content)
            threads = [
                executor.submit(
                    parse_gcs_zip, zip_file, path, anisotropy, min_size
                )
                for path in paths
            ]

    # Process results
    valid_labels = dict()
    for thread in as_completed(threads):
        valid_labels.update(thread.result())
    return valid_labels


def parse_gcs_zip(zip_file, path, anisotropy, min_size):
    """
    Reads swc file stored at "path" which points to a file in a GCS bucket.

    Parameters
    ----------
    zip_file : ZipFile
        Zip file containing swc file to be read.
    path : str
        Path to swc file to be read.
    anisotropy : list[float]
        Image to World scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.
    min_size : int
        Threshold on the number of nodes contained in an swc file. Only swc
        files with more than "min_size" nodes are stored in "valid_labels".

    Returns
    -------
    list
        Entries of an swc file.

    """
    contents = read_from_cloud(zip_file, path)
    if len(contents) > min_size:
        swc_id = int(utils.get_id(path))
        return {swc_id: get_coords(contents, anisotropy)}
    else:
        return dict()


def read_from_local(path):
    """
    Reads swc file stored at "path" on local machine.

    Parameters
    ----------
    path : str
        Path to swc file to be read.

    Returns
    -------
    list
        List such that each entry is a line from the swc file.

    """
    with open(path, "r") as file:
        return file.readlines()


def read_from_cloud(zip_file, path):
    """
    Reads the content of an swc file from a zip file in a GCS bucket.

    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8").splitlines()


def get_coords(contents, anisotropy):
    """
    Gets the xyz coords from the swc file at "path".

    Parameters
    ----------
    path : str
        Path to swc file to be parsed.
    anisotropy : list[float]
        Image to World scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    numpy.ndarray
        xyz coords from an swc file.

    """
    coords_list = []
    offset = [0, 0, 0]
    for line in contents:
        if line.startswith("# OFFSET"):
            parts = line.split()
            offset = read_xyz(parts[2:5])
        if not line.startswith("#"):
            parts = line.split()
            coord = read_xyz(parts[2:5], anisotropy=anisotropy, offset=offset)
            coords_list.append(coord)
    return np.array(coords_list)


def read_xyz(xyz, anisotropy=[1.0, 1.0, 1.0], offset=[0, 0, 0]):
    """
    Reads the (x,y,z) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    coord : str
        xyz coordinate.
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
    graph = nx.Graph(swc_id=utils.get_id(path))
    offset = [0, 0, 0]
    for line in read_from_local(path):
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
