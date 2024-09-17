# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from io import BytesIO, StringIO
from zipfile import ZipFile

import networkx as nx
import numpy as np
from google.cloud import storage
from tqdm import tqdm

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import utils

ANISOTROPY = [0.748, 0.748, 1.0]


class Reader:
    """
    Class that reads swc files that are stored as (1) local directory of swcs,
    (2) gcs directory of zips containing swcs, (3) local zip containing swcs.

    """

    def __init__(
        self, anisotropy=[1.0, 1.0, 1.0], min_size=0, return_graphs=False
    ):
        """
        Initializes a Reader object that loads swc files.

        Parameters
        ----------
        anisotropy : list[float], optional
            Image to world scaling factors applied to xyz coordinates to
            account for anisotropy of the microscope. The default is
            [1.0, 1.0, 1.0].
        min_size : int, optional
            Threshold on the number of nodes in swc file. Only swc files with
            more than "min_size" nodes are stored in "xyz_coords". The default
            is 0.
        return_graphs : bool, optional
            Indication of whether to return contents of swc file in the form
            of a graph. The default is False.

        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.min_size = min_size
        self.return_graphs = return_graphs

    def load(self, swc_pointer):
        """
        Load data based on the type and format of the provided "swc_pointer".

        Parameters
        ----------
        swc_pointer : dict, list, str
            Must be one of the following: (1) A dictionary for loading from a
            GCS bucket, (2) list of file paths for loading from local paths,
            (3) path to one swc file, or (4) path to ".zip" file containing
            swc files.

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the xyz coordinates or graph read
            from that swc file.

        """
        if type(swc_pointer) is dict:
            return self.load_from_gcs(swc_pointer)
        elif type(swc_pointer) is list:
            return self.load_from_local_paths(swc_pointer)
        elif type(swc_pointer) is str and ".zip" in swc_pointer:
            return self.load_from_local_zip(swc_pointer)
        elif type(swc_pointer) is str and ".swc" in swc_pointer:
            return self.load_from_local_path(swc_pointer)
        else:
            print("SWC Pointer is not Valid!")

    def load_from_local_paths(self, swc_paths):
        """
        Reads swc files from local machine, then returns either the xyz
        coordinates or graphs.

        Paramters
        ---------
        swc_paths : list
            List of paths to swc files stored on the local machine.

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the xyz coordinates or graph read
            from that swc file.

        """
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = list()
            for path in swc_paths:
                processes.append(
                    executor.submit(self.load_from_local_path, path)
                )

            # Store results
            swc_dicts = dict()
            for process in as_completed(processes):
                swc_dicts.update(process.result())
        return swc_dicts

    def load_from_local_path(self, path):
        """
        Reads a single swc file from local machine, then returns either the
        xyz coordinates or graphs.

        Paramters
        ---------
        path : str
            Path to swc file stored on the local machine.

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the xyz coordinates or graph read
            from that swc file.

        """
        content = utils.read_txt(path)
        if len(content) > self.min_size:
            key = utils.get_id(path)
            if self.return_graphs:
                result = self.get_graph(content)
                result.graph["filename"] = os.path.basename(path)
            else:
                result = self.get_coords(content)
            return {key: result}
        else:
            return dict()

    def load_from_local_zip(self, zip_path):
        """
        Reads swc files from zip on the local machine, then returns either the
        xyz coordinates or graph. Note this routine is hard coded for computing
        projected run length.

        Paramters
        ---------
        swc_paths : list or dict
            If swc files are on local machine, list of paths to swc files where
            each file corresponds to a neuron in the prediction. If swc files
            are on cloud, then dict with keys "bucket_name" and "path".

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the the xyz coordinates read from
            that swc file.

        """
        swc_dict = dict()
        with ZipFile(zip_path, "r") as zip:
            swc_files = [f for f in zip.namelist() if f.endswith(".swc")]
            for f in tqdm(swc_files, desc="Loading Fragments"):
                # Check whether to store content
                content = utils.read_zip(zip, f).splitlines()
                if len(content) > self.min_size:
                    key = utils.get_id(f)
                    if self.return_graphs:
                        swc_dict[key] = self.get_graph(content)
                        swc_dict[key].graph["filename"] = f
                    else:
                        swc_dict[key] = self.get_coords(content)
        return swc_dict

    def load_from_gcs(self, gcs_dict):
        """
        Reads swc files from a GCS bucket.

        Parameters
        ----------
        gcs_dict : dict
            Dictionary where keys are "bucket_name" and "path".

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the the xyz coordinates read from
            that swc file.

        """
        # Initializations
        bucket = storage.Client().bucket(gcs_dict["bucket_name"])
        zip_paths = utils.list_gcs_filenames(bucket, gcs_dict["path"], ".zip")
        print("Downloading swc files from cloud...")
        print("# zip files:", len(zip_paths))

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = []
            for path in zip_paths:
                zip_content = bucket.blob(path).download_as_bytes()
                processes.append(
                    executor.submit(self.load_from_cloud_zip, zip_content)
                )

            # Store results
            swc_dicts = dict()
            for process in tqdm(as_completed(processes)):
                swc_dicts.update(process.result())
        return swc_dicts

    def load_from_cloud_zip(self, zip_content):
        """
        Reads swc files from a zip that has been downloaded from a cloud
        bucket.

        Parameters
        ----------
        zip_content : ...
            content of a zip file.

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the the xyz coordinates read from
            that swc file.

        """
        with ZipFile(BytesIO(zip_content)) as zip_file:
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = []
                for f in utils.list_files_in_zip(zip_content):
                    threads.append(
                        executor.submit(
                            self.load_from_cloud_zipped_file, zip_file, f
                        )
                    )

                # Process results
                swc_dicts = dict()
                for thread in as_completed(threads):
                    swc_dicts.update(thread.result())
        return swc_dicts

    def load_from_cloud_zipped_file(self, zip_file, path):
        """
        Reads swc file stored at "path" which points to a file in a zip.

        Parameters
        ----------
        zip_file : ZipFile
            Zip containing swc file to be read.
        path : str
            Path to swc file to be read.

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the the xyz coordinates or graph
            read from that swc file.

        """
        content = utils.read_zip(zip_file, path).splitlines()
        if len(content) > self.min_size:
            key = utils.get_id(path)
            if self.return_graphs:
                graph = self.get_graph(content)
                graph.graph["filename"] = os.path.basename(path)
                return {key: graph}
            else:
                return {key: self.get_coords(content)}
        else:
            return dict()

    def get_coords(self, content):
        """
        Gets the xyz coords from the an swc file that has been read and stored
        as "content".

        Parameters
        ----------
        content : list[str]
            Entries from swc where each item is the text string from an swc.
        anisotropy : list[float]
            Image to world scaling factors applied to xyz coordinates to
            account for anisotropy of the microscope.

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
                offset = self.read_xyz(parts[2:5], offset)
            if not line.startswith("#"):
                parts = line.split()
                coords_list.append(self.read_xyz(parts[2:5], offset))
        return np.array(coords_list)

    def read_xyz(self, xyz_str, offset):
        """
        Reads the coordinates from a string, then transforms them to image
        coordinates (if applicable).

        Parameters
        ----------
        xyz_str : str
            Coordinate stored in a str.
        offset : list[int]
            Offset of coordinates in swc file.

        Returns
        -------
        numpy.ndarray
            xyz coordinates of an entry from an swc file.

        """
        xyz = np.zeros((3))
        for i in range(3):
            xyz[i] = self.anisotropy[i] * (float(xyz_str[i]) + offset[i])
        return np.round(xyz).astype(int)

    def get_graph(self, content):
        """
        Reads an swc file and builds an undirected graph from it.

        Parameters
        ----------
        path : str
            Path to swc file to be read.

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
                offset = self.read_xyz(parts[2:5])
            if not line.startswith("#"):
                parts = line.split()
                child = int(parts[0])
                parent = int(parts[-1])
                xyz = self.read_xyz(parts[2:5], offset=offset)
                graph.add_node(child, xyz=xyz)
                if parent != -1:
                    graph.add_edge(parent, child)

        # Set graph-level attributes
        graph.graph["n_edges"] = graph.number_of_edges()
        graph.graph["run_length"] = gutils.compute_run_length(graph)
        return graph


# -- write --
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
    x, y, z = tuple(utils.to_world(xyz))
    entry = f"{node_id} 2 {x} {y} {z} 8 {parent_id}"
    return entry


def to_zipped_swc(zip_writer, graph, color=None):
    """
    Writes a graph to an swc file that is to be stored in a zip.

    Parameters
    ----------
    zip_writer : zipfile.ZipFile
        ...
    graph : networkx.Graph
        Graph to be written to an swc file.
    color : str, optional
        ...

    Returns
    -------
    None

    """
    with StringIO() as text_buffer:
        # Preamble
        n_entries = 0
        node_to_idx = dict()
        if color:
            text_buffer.write("# COLOR " + color)
        text_buffer.write("# id, type, z, y, x, r, pid")

        # Write entries
        r = 5 if color else 3
        for i, j in nx.dfs_edges(graph):
            # Special Case: Root
            x, y, z = tuple(utils.to_world(graph.nodes[i]["xyz"]))
            if n_entries == 0:
                parent = -1
                node_to_idx[i] = 1
                text_buffer.write("\n" + f"1 2 {x} {y} {z} {r} {parent}")
                n_entries += 1

            # General Case
            node = n_entries + 1
            parent = node_to_idx[i]
            node_to_idx[j] = n_entries + 1
            text_buffer.write("\n" + f"{node} 2 {x} {y} {z} {r} {parent}")
            n_entries += 1

        # Finish
        zip_writer.writestr(graph.graph["filename"], text_buffer.getvalue())
