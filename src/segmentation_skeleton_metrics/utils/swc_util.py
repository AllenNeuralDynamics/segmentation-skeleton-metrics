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

from collections import deque
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from io import StringIO
from tqdm import tqdm
from zipfile import ZipFile

import networkx as nx
import numpy as np
import os

from segmentation_skeleton_metrics.utils import img_util, util


class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, (3) local directory of ZIP archives or (4) GCS directory of ZIP
    archives.

    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), selected_ids=None):
        """
        Initializes a Reader object that loads swc files.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to world scaling factors applied to xyz coordinates to
            account for anisotropy of the microscope. The default is
            (1.0, 1.0, 1.0).

        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.selected_ids = selected_ids or set()

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

        Returns
        -------
        dict
            Dictionary whose keys are filnames of SWC files and values are the
            corresponding graphs.

        """
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
        if self.confirm_load(filename):
            return self.parse(content, filename)
        else:
            return None

    def load_from_local_paths(self, paths):
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
            pbar = tqdm(total=len(paths), desc="Read SWCs")
            for path in paths:
                filename = os.path.basename(path)
                if self.confirm_load(filename):
                    threads.append(
                        executor.submit(self.load_from_local_path, path)
                    )

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                swc_dicts.append(thread.result())
                pbar.update(1)
        return swc_dicts

    def load_from_local_zips(self, zip_dir):
        # Initializations
        zip_names = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]
        pbar = tqdm(total=len(zip_names), desc="Read SWCs")

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign threads
            processes = list()
            for f in zip_names:
                zip_path = os.path.join(zip_dir, f)
                processes.append(
                    executor.submit(self.load_from_local_zip, zip_path)
                )

            # Store results
            swc_dicts = deque()
            for process in as_completed(processes):
                swc_dicts.extend(process.result())
                pbar.update(1)
        return swc_dicts

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
            filesnames = [f for f in zipfile.namelist() if f.endswith(".swc")]
            for filename in filesnames:
                if self.confirm_load(filename):
                    threads.append(
                        executor.submit(
                            self.load_from_zipped_file, zipfile, filename
                        )
                    )

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                swc_dicts.append(thread.result())
        return swc_dicts

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
        return self.parse(content, filename)

    def confirm_load(self, filename):
        if len(self.selected_ids) > 0:
            segment_id = util.get_segment_id(filename)
            return True if segment_id in self.selected_ids else False
        else:
            return True

    # -- Process Text ---
    def parse(self, content, filename):
        """
        Parses an SWC file to extract the content which is stored in a dict.
        Note that node_ids from SWC are reindex from 0 to n-1 where n is the
        number of nodes in the SWC file.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        dict
            Dictionaries whose keys and values are the attribute names
            and values from an SWC file.

        """
        # Initializations
        swc_id, _ = os.path.splitext(filename)
        content, offset = self.process_content(content)
        swc_dict = {
            "id": np.zeros((len(content)), dtype=int),
            "pid": np.zeros((len(content)), dtype=int),
            "voxel": np.zeros((len(content), 3), dtype=np.int32),
            "swc_id": swc_id,
        }

        # Parse content
        for i, line in enumerate(content):
            parts = line.split()
            swc_dict["id"][i] = parts[0]
            swc_dict["pid"][i] = parts[-1]
            swc_dict["voxel"][i] = self.read_voxel(parts[2:5], offset)
        return swc_dict

    def process_content(self, content):
        """
        Processes lines of text from an SWC file, extracting an offset
        value and returning the remaining content starting from the line
        immediately after the last commented line.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        List[str]
            A list of strings representing the lines of text starting from the
            line immediately after the last commented line.
        List[int]
            Offset used to shift coordinates.

        """
        offset = (0, 0, 0)
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = self.read_voxel(parts[2:5])
            if not line.startswith("#"):
                return content[i:], offset

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
        Tuple[int]
            xyz coordinates of an entry from an SWC file.

        """
        xyz = [float(xyz_str[i]) + offset[i] for i in range(3)]
        return img_util.to_voxels(xyz, self.anisotropy)


# --- Helpers ---
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
            x, y, z = tuple(img_util.to_physical(graph.nodes[i]["voxel"]))
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
