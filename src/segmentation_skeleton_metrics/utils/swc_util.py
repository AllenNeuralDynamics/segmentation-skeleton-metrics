# -*- coding: utf-8 -*-
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with SWC files.

An SWC file is a text-based file format used to represent the directed
graphical structure of a neuron. It contains a series of nodes such that each
has the following attributes:
    "id" (int): node ID
    "type" (int): node type (e.g. soma, axon, dendrite)
    "x" (float): x coordinate
    "y" (float): y coordinate
    "z" (float): z coordinate
    "pid" (int): node ID of parent

Note: Each uncommented line in an SWC file corresponds to a node and contains
      these attributes in the same order.

"""


from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from google.cloud import storage
from io import BytesIO, StringIO
from tqdm import tqdm
from zipfile import ZipFile

import networkx as nx
import os

from segmentation_skeleton_metrics.utils import graph_util as gutil, util


class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, (3) local directory of ZIP archives or (4) GCS directory of ZIP
    archives.

    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), min_size=0):
        """
        Initializes a Reader object that loads swc files.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to world scaling factors applied to xyz coordinates to
            account for anisotropy of the microscope. The default is
            (1.0, 1.0, 1.0).
        min_size : int, optional
            Threshold on the number of nodes in swc file. Only swc files with
            more than "min_size" nodes are returned.
        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.min_size = min_size

    # --- Load Data ---
    def load(self, swc_pointer):
        """
        Load SWCs files based on the type pointer provided.

        Parameters
        ----------
        swc_pointer : dict, list, str
            Object that points to SWC files to be read, must be one of:
                - swc_dir (str): Path to directory containing SWC files.
                - swc_path (str): Path to single SWC file.
                - swc_path_list (List[str]): List of paths to SWC files.
                - swc_zip (str): Path to a ZIP archive containing SWC files.
                - swc_zip_dir (str): Path to directory of ZIPs with SWC files.
                - gcs_dict (dict): Dictionary that contains the keys
                  "bucket_name" and "path" to read from a GCS bucket.

        Returns
        -------
        dict
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
        # GCS bucket containing ZIP archives with SWC files
        if isinstance(swc_pointer, dict):
            return self.load_from_gcs(swc_pointer)

        # List of paths to SWC files
        if isinstance(swc_pointer, list):
            return self.load_from_local_paths(swc_pointer)

        # Directory containing...
        if os.path.isdir(swc_pointer):
            # ZIP archives with SWC files
            paths = util.list_paths(swc_pointer, extension=".zip")
            if len(paths) > 0:
                return self.load_from_local_zips(swc_pointer)

            # SWC files
            paths = util.list_paths(swc_pointer, extension=".swc")
            if len(paths) > 0:
                return self.load_from_local_paths(paths)

            raise Exception("Directory is invalid!")

        # Path to...
        if isinstance(swc_pointer, str):
            # ZIP archive with SWC files
            if ".zip" in swc_pointer:
                return self.load_from_local_zip(swc_pointer)

            # Path to single SWC file
            if ".swc" in swc_pointer:
                return self.load_from_local_path(swc_pointer)

            raise Exception("Path is invalid!")

        raise Exception("SWC Pointer is inValid!")

    def load_from_local_paths(self, swc_paths):
        """
        Reads list of SWC files stored on the local machine.

        Paramters
        ---------
        swc_paths : list
            List of paths to SWC files stored on the local machine.

        Returns
        -------
        dict
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for path in swc_paths:
                threads.append(
                    executor.submit(self.load_from_local_path, path)
                )

            # Store results
            graph_dict = dict()
            pbar = tqdm(total=len(threads), desc="Load SWCs")
            for thread in as_completed(threads):
                graph_dict.update(thread.result())
                pbar.update(1)
        return graph_dict

    def load_from_local_path(self, path):
        """
        Reads a single SWC file from local machine.

        Paramters
        ---------
        path : str
            Path to SWC file stored on the local machine.

        Returns
        -------
        dict
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
        content = util.read_txt(path)
        filename = os.path.basename(path)
        return self.process_content(content, filename)

    def load_from_local_zips(self, zip_dir):
        filenames = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]
        pbar = tqdm(total=len(filenames), desc="Load SWCs")
        with ProcessPoolExecutor() as executor:
            # Assign threads
            processes = list()
            for f in filenames:
                zip_path = os.path.join(zip_dir, f)
                processes.append(
                    executor.submit(self.load_from_local_zip, zip_path, False)
                )

            # Store results
            graph_dict = dict()
            for process in as_completed(processes):
                graph_dict.update(process.result())
                pbar.update(1)
        return graph_dict

    def load_from_local_zip(self, zip_path):
        """
        Reads SWC files from zip on the local machine.

        Paramters
        ---------
        swc_paths : list or dict
            If swc files are on local machine, list of paths to swc files where
            each file corresponds to a neuron in the prediction. If swc files
            are on cloud, then dict with keys "bucket_name" and "path".

        Returns
        -------
        dict
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            zipfile = ZipFile(zip_path, "r")
            for f in  [f for f in zipfile.namelist() if f.endswith(".swc")]:
                threads.append(
                    executor.submit(self.load_from_zipped_file, zipfile, f)
                )

            # Store results
            graph_dict = dict()
            for thread in as_completed(threads):
                graph_dict.update(thread.result())
        return graph_dict

    def load_from_zipped_file(self, zipfile, path):
        """
        Reads swc file stored at "path" which points to a file in a zip.

        Parameters
        ----------
        zipfile : ZipFile
            Zip containing swc file to be read.
        path : str
            Path to swc file to be read.

        Returns
        -------
        dict
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
        content = util.read_zip(zipfile, path).splitlines()
        filename = os.path.basename(path)
        return self.process_content(content, filename)

    def load_from_gcs(self, gcs_dict):
        """
        Reads ZIP archives containing SWC files stored in a GCS bucket.

        Parameters
        ----------
        gcs_dict : dict
            Dictionary with the keys are "bucket_name" and "path".

        Returns
        -------
        dict
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
        # Initializations
        bucket = storage.Client().bucket(gcs_dict["bucket_name"])
        zip_paths = util.list_gcs_filenames(bucket, gcs_dict["path"], ".zip")

        # Main
        pbar = tqdm(total=len(zip_paths), desc="Download Fragments")
        with ProcessPoolExecutor(max_workers=1) as executor:
            # Assign processes
            processes = []
            for path in zip_paths:
                zip_content = bucket.blob(path).download_as_bytes()
                processes.append(
                    executor.submit(self.load_from_cloud_zip, zip_content)
                )

            # Store results
            graph_dict = dict()
            for process in as_completed(processes):
                graph_dict.update(process.result())
                pbar.update(1)
        return graph_dict

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
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
        with ZipFile(BytesIO(zip_content)) as zipfile:
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = []
                for f in util.list_files_in_zip(zip_content):
                    threads.append(
                        executor.submit(
                            self.load_from_zipped_file, zipfile, f
                        )
                    )

                # Process results
                graph_dict = dict()
                for thread in as_completed(threads):
                    graph_dict.update(thread.result())
        return graph_dict

    # -- Process Data ---
    def process_content(self, content, filename):
        graph = self.get_graph(content)
        if graph is not None:
            graph.graph["filename"] = filename
            name, _ = os.path.splitext(filename)
            return {name: graph}
        return dict()

    def read_voxel(self, xyz_str, offset):
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
        xyz = [float(xyz_str[i]) + offset[i] for i in range(3)]
        return util.to_voxels(xyz, self.anisotropy)

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
            # Check for offset
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = self.read_voxel(parts[2:5])

            # Check for entry
            if not line.startswith("#"):
                parts = line.split()
                child = int(parts[0])
                parent = int(parts[-1])
                voxel = self.read_voxel(parts[2:5], offset=offset)
                graph.add_node(child, voxel=voxel)
                if parent != -1:
                    graph.add_edge(parent, child)

        # Set graph-level attributes
        graph.graph["n_edges"] = graph.number_of_edges()
        graph.graph["run_length"] = gutil.compute_run_length(graph)
        if graph.graph["run_length"] > self.min_size:
            return graph
        else:
            return None


# --- Write ---
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
            x, y, z = tuple(util.to_world(graph.nodes[i]["voxel"]))
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
