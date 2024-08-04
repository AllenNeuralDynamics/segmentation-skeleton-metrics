# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from io import BytesIO
from zipfile import ZipFile

import networkx as nx
import numpy as np
from google.cloud import storage

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import utils


def init_valid_labels(swc_paths, min_size, anisotropy=[1.0, 1.0, 1.0]):
    """
    Reads swc files and extracts the xyz coordinates.

    Paramters
    ---------
    swc_paths : list or dict
        If swc files are on local machine, list of paths to swc files where
        each file corresponds to a neuron in the prediction. If swc files are
        on cloud, then dict with keys "bucket_name" and "path".
    min_size : int
        Threshold on the number of nodes stored in an swc file. Only swc files
        with more than "min_size" nodes are stored in "swc_coords".
   anisotropy : list[float], optional
       Image to world scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    dict
        Dictionary where each item is an swc_id and an array of the xyz
        coordinates read from cooresponding swc file.

    """
    if type(swc_paths) is list:
        return set(parse_local_paths(swc_paths, min_size, anisotropy).keys())
    elif type(swc_paths) is dict:
        return set(parse_cloud_paths(swc_paths, min_size, anisotropy).keys())
    else:
        return None


def parse_local_paths(swc_paths, min_size, anisotropy):
    """
    Reads swc files from local machine and extracts the xyz coordinates.
    --> hard coded

    Paramters
    ---------
    swc_paths : list or dict
        If swc files are on local machine, list of paths to swc files where
        each file corresponds to a neuron in the prediction. If swc files are
        on cloud, then dict with keys "bucket_name" and "path".
    min_size : int
        Threshold on the number of nodes stored in an swc file. Only swc files
        with more than "min_size" nodes are stored in "swc_coords".
   anisotropy : list[float], optional
       Image to world scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    dict
        Dictionary that maps an swc_id to the the xyz coordinates read from
        that swc file.

    """
    graphs = dict()
    for path in swc_paths:
        content = utils.read_txt(path)
        if len(content) > min_size:
            key = int(utils.get_id(path))
            graphs[key] = get_coords(content, anisotropy)
    return graphs


def parse_local_zip(zip_path, min_size, anisotropy=[1.0, 1.0, 1.0]):
    """
    Reads swc files from a zip stored on the local machine and extracts the
    xyz coordinates.

    Paramters
    ---------
    swc_paths : list or dict
        If swc files are on local machine, list of paths to swc files where
        each file corresponds to a neuron in the prediction. If swc files are
        on cloud, then dict with keys "bucket_name" and "path".
    min_size : int
        Threshold on the number of nodes stored in an swc file. Only swc files
        with more than "min_size" nodes are stored in "swc_coords".
   anisotropy : list[float], optional
       Image to world scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    dict
        Dictionary that maps an swc_id to the the xyz coordinates read from
        that swc file.

    """
    graphs = dict()
    anisotropy = [1.0 / 0.748, 1.0 / 0.748, 1.0]  # hard coded
    with ZipFile(zip_path, "r") as zip_file:
        files = zip_file.namelist()
        for swc_file in [f for f in files if f.endswith(".swc")]:
            content = utils.read_zip(zip_file, swc_file).splitlines()
            if len(content) > min_size:
                key = int(utils.get_id(swc_file))
                graphs[key] = to_graph(content, anisotropy=anisotropy)
    return graphs


def parse_cloud_paths(cloud_dict, min_size, anisotropy):
    """
    Reads swc files from a GCS bucket and extracts the xyz coordinates.

    Parameters
    ----------
    cloud_dict : dict
        Dictionary where keys are "bucket_name" and "path".
    min_size : int
        Threshold on the number of nodes stored in an swc file. Only swc files
        with more than "min_size" nodes are stored in "swc_coords".
   anisotropy : list[float], optional
       Image to world scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    dict
        Dictionary that maps an swc_id to the the xyz coordinates read from
        that swc file.

    """
    # Initializations
    bucket = storage.Client().bucket(cloud_dict["bucket_name"])
    zip_paths = utils.list_gcs_filenames(bucket, cloud_dict["path"], ".zip")
    print("Downloading predicted swc files from cloud...")
    print("# zip files:", len(zip_paths))

    # Main
    cnt = 1
    chunk_size = len(zip_paths) * 0.02
    t0, t1 = utils.init_timers()
    with ProcessPoolExecutor() as executor:
        # Assign processes
        processes = []
        for i, path in enumerate(zip_paths):
            zip_content = bucket.blob(path).download_as_bytes()
            processes.append(
                executor.submit(download, zip_content, min_size, anisotropy)
            )
            if i >= cnt * chunk_size:
                cnt, t1 = utils.report_progress(
                    i, len(zip_paths), chunk_size, cnt, t0, t1
                )

        # Store results
        swc_coords = dict()
        for i, process in enumerate(as_completed(processes)):
            swc_coords.update(process.result())
    return swc_coords


def download(zip_content, min_size, anisotropy):
    """
    Downloads the content from each swc file contained in the zip file at
    "zip_path".

    Parameters
    ----------
    zip_content : ...
        content of a zip file.
    min_size : int
        Threshold on the number of nodes stored in an swc file. Only swc files
        with more than "min_size" nodes are stored in "swc_coords".
   anisotropy : list[float], optional
       Image to world scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    dict
        Dictionary that maps an swc_id to the the xyz coordinates read from
        that swc file.

    """
    with ZipFile(BytesIO(zip_content)) as zip_file:
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = [
                executor.submit(
                    parse_gcs_zip, zip_file, f, min_size, anisotropy
                )
                for f in utils.list_files_in_zip(zip_content)
            ]

            # Process results
            swc_coords = dict()
            for thread in as_completed(threads):
                swc_coords.update(thread.result())
    return swc_coords


def parse_gcs_zip(zip_file, path, min_size, anisotropy):
    """
    Reads swc file stored at "path" which points to a file in a GCS bucket.

    Parameters
    ----------
    zip_file : ZipFile
        Zip containing swc file to be read.
    path : str
        Path to swc file to be read.
    min_size : int
        Threshold on the number of nodes stored in an swc file. Only swc files
        with more than "min_size" nodes are stored in "swc_coords".
    anisotropy : list[float], optional
       Image to world scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    dict
        Dictionary that maps an swc_id to the the xyz coordinates read from
        that swc file.

    """
    content = utils.read_zip(zip_file, path).splitlines()
    if len(content) > min_size:
        return {int(utils.get_id(path)): get_coords(content, anisotropy)}
    else:
        return dict()


def get_coords(content, anisotropy):
    """
    Gets the xyz coords from the an swc file that has been read and stored as
    "content".

    Parameters
    ----------
    content : list[str]
        Entries in swc file where each entry is the text string from an swc.
    anisotropy : list[float]
        Image to world scaling factors applied to xyz coordinates to account
        for anisotropy of the microscope.

    Returns
    -------
    numpy.ndarray
        xyz coords from an swc file.

    """
    coords_list = []
    offset = [0, 0, 0]
    for line in content:
        if line.startswith("# OFFSET"):
            parts = line.split()
            offset = read_xyz(parts[2:5], anisotropy, offset)
        if not line.startswith("#"):
            parts = line.split()
            coords_list.append(read_xyz(parts[2:5], anisotropy, offset))
    return np.array(coords_list)


def read_xyz(xyz, anisotropy, offset):
    """
    Reads the xyz coordinates from an swc file, then transforms the
    coordinates with respect to "anisotropy" and "offset".

    Parameters
    ----------
    xyz : str
        xyz coordinate stored in a str.
    anisotropy : list[float]
        Image to real-world coordinates scaling factors applied to "xyz".
    offset : list[int]
        Offset of xyz coordinates in swc file.

    Returns
    -------
    numpy.ndarray
        xyz coordinates of an entry from an swc file.

    """
    xyz = [float(xyz[i]) + offset[i] for i in range(3)]
    return np.array([xyz[i] * anisotropy[i] for i in range(3)], dtype=int)


def save(path, xyz_1, xyz_2, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path : str
        Path on local machine that swc file will be written to.
    xyz_1 : ...
        ...
    xyz_2 : ...
        ...
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


def to_graph(content, anisotropy=[1.0, 1.0, 1.0]):
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
    # Build Gaph
    graph = nx.Graph()
    offset = [0, 0, 0]
    for line in content:
        if line.startswith("# OFFSET"):
            parts = line.split()
            offset = read_xyz(parts[2:5], anisotropy)
        if not line.startswith("#"):
            parts = line.split()
            child = int(parts[0])
            parent = int(parts[-1])
            xyz = read_xyz(parts[2:5], anisotropy, offset=offset)
            graph.add_node(child, xyz=xyz)
            if parent != -1:
                graph.add_edge(parent, child)

    # Set graph-level attributes
    graph.graph["initial_number_of_edges"] = graph.number_of_edges()
    graph.graph["initial_run_length"] = gutils.compute_run_length(graph)
    return graph
