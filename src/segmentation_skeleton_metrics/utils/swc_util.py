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
    as_completed,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from google.cloud import storage
from io import BytesIO, StringIO
from tqdm import tqdm
from zipfile import ZipFile

import numpy as np
import os

from segmentation_skeleton_metrics.utils import img_util, util


class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, and (3) local directory of ZIP archives.

    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), selected_ids=None):
        """
        Initializes a Reader object that reads SWC files.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        selected_ids : Set[int], optional
            Only SWC files with an swc_id contained in this set are read. The
            default is None.

        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.selected_ids = selected_ids or set()

    # --- Read Data ---
    def read(self, swc_pointer):
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
        Deque[dict]
            List of dictionaries whose keys and values are the attribute names
            and values from the SWC files. Each dictionary contains the
            following items:
                - "id": unique identifier of each node in an SWC file.
                - "pid": parent ID of each node.
                - "radius": radius value corresponding to each node.
                - "xyz": coordinate corresponding to each node.
                - "filename": filename of SWC file
                - "swc_id": name of SWC file, minus the ".swc".

        """
        # Dictionary with GCS specs
        if isinstance(swc_pointer, dict):
            return self.read_from_gcs(swc_pointer)

        # List of paths to SWC files
        if isinstance(swc_pointer, list):
            return self.read_from_paths(swc_pointer)

        # Directory containing...
        if os.path.isdir(swc_pointer):
            # ZIP archives with SWC files
            paths = util.list_paths(swc_pointer, extension=".zip")
            if len(paths) > 0:
                return self.read_from_zips(swc_pointer)

            # SWC files
            paths = util.read_paths(swc_pointer, extension=".swc")
            if len(paths) > 0:
                return self.read_from_paths(paths)

            raise Exception("Directory is invalid!")

        # Path to...
        if isinstance(swc_pointer, str):
            # ZIP archive with SWC files
            if ".zip" in swc_pointer:
                return self.read_from_zip(swc_pointer)

            # Path to single SWC file
            if ".swc" in swc_pointer:
                return self.read_from_path(swc_pointer)

            raise Exception("Path is invalid!")

        raise Exception("SWC Pointer is inValid!")

    def read_from_path(self, path):
        """
        Reads a single SWC file.

        Paramters
        ---------
        path : str
            Path to SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.

        """
        content = util.read_txt(path)
        filename = os.path.basename(path)
        if self.confirm_read(filename):
            return self.parse(content, filename)
        else:
            return None

    def read_from_paths(self, paths):
        """
        Reads SWC files given a list of paths.

        Paramters
        ---------
        swc_paths : List[str]
            Paths to SWC files.

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            pbar = tqdm(total=len(paths), desc="Read SWCs")
            for path in paths:
                filename = os.path.basename(path)
                if self.confirm_read(filename):
                    threads.append(
                        executor.submit(self.read_from_path, path)
                    )

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
                pbar.update(1)
        return swc_dicts

    def read_from_zips(self, zip_dir):
        """
        Processes a directory containing ZIP archives with SWC files.

        Parameters
        ----------
        zip_dir : str
            Path to directory containing ZIP archives with SWC files.

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.

        """
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
                    executor.submit(self.read_from_zip, zip_path)
                )

            # Store results
            swc_dicts = deque()
            for process in as_completed(processes):
                swc_dicts.extend(process.result())
                pbar.update(1)
        return swc_dicts

    def read_from_zip(self, zip_path):
        """
        Reads SWC files from a ZIP archive.

        Paramters
        ---------
        zip_path : str
            Path to ZIP archive.

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            zipfile = ZipFile(zip_path, "r")
            filenames = [f for f in zipfile.namelist() if f.endswith(".swc")]
            for filename in filenames:
                if self.confirm_read(filename):
                    threads.append(
                        executor.submit(
                            self.read_from_zipped_file, zipfile, filename
                        )
                    )

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    def read_from_zipped_file(self, zipfile, path):
        """
        Reads an SWC file stored in a ZIP archive.

        Parameters
        ----------
        zip_file : ZipFile
            ZIP archive containing SWC files.
        path : str
            Path to SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.

        """
        content = util.read_zip(zipfile, path).splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename) if len(content) > 40 else None

    def read_from_gcs(self, gcs_dict):
        """
        Reads SWC files from ZIP archives stored in a GCS bucket.

        Parameters
        ----------
        gcs_dict : dict
            Dictionary with the keys "bucket_name" and "path" that specify
            where the ZIP archives are located in a GCS bucket.

        Returns
        -------
        Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.

        """
        # List filenames
        swc_paths = util.list_gcs_filenames(gcs_dict, ".swc")
        zip_paths = util.list_gcs_filenames(gcs_dict, ".zip")

        # Call reader
        if len(swc_paths) > 0:
            return self.read_from_gcs_swcs(gcs_dict["bucket_name"], swc_paths)
        if len(zip_paths) > 0:
            return self.read_from_gcs_zips(gcs_dict["bucket_name"], zip_paths)

        # Error
        raise Exception(f"GCS Pointer is invalid -{gcs_dict}-")

    def read_from_gcs_swcs(self, bucket_name, swc_paths):
        pbar = tqdm(total=len(swc_paths), desc="Read SWCs")
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for path in swc_paths:
                threads.append(
                    executor.submit(self.read_from_gcs_swc, bucket_name, path)
                )

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                swc_dicts.append(thread.result())
                pbar.update(1)
        print("# swcs:", len(swc_dicts))
        return swc_dicts

    def read_from_gcs_swc(self, bucket_name, path):
        # Initialize cloud reader
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path)

        # Parse swc contents
        content = blob.download_as_text().splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename)

    def read_from_gcs_zips(self, bucket_name, zip_paths):
        swc_dicts = deque()
        for zip_path in tqdm(zip_paths, desc="Read SWCs"):
            swc_dicts.extend(self.read_from_gcs_zip(bucket_name, zip_path))
        return swc_dicts

    def read_from_gcs_zip(self, bucket_name, path):
        """
        Reads SWC files stored in a ZIP archive downloaded from a GCS
        bucket.

        Parameters
        ----------
        zip_content : bytes
            Content of a ZIP archive.

        Returns
        -------
        Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.


        """
        # Initialize cloud reader
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Parse Zip
        swc_dicts = deque()
        zip_content = bucket.blob(path).download_as_bytes()
        with ZipFile(BytesIO(zip_content), "r") as zip_file:
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = list()
                for filename in zip_file.namelist():
                    if self.confirm_read(filename):
                        threads.append(
                            executor.submit(
                                self.read_from_zipped_file, zip_file, filename
                            )
                        )

                # Process results
                for thread in as_completed(threads):
                    swc_dicts.append(thread.result())
        return swc_dicts

    def confirm_read(self, filename):
        """
        Checks whether the swc_id corresponding to the given filename is
        contained in the attribute "selected_ids".

        Parameters
        ----------
        filename : str
            Name of SWC file to be checked.

        Returns
        -------
        bool
            Indication of whether to read SWC file.

        """
        if self.selected_ids:
            segment_id = util.get_segment_id(filename)
            return True if segment_id in self.selected_ids else False
        else:
            return True

    # -- Process Text ---
    def parse(self, content, filename):
        """
        Parses an SWC file to extract the content which is stored in a dict.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.

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
        tuple
            A tuple containing the following:
            - "content" (List[str]): lines from an SWC file after comments.
            - "offset" (Tuple[int]): offset used to shift coordinate.

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
        Reads a coordinate from a string and converts it to voxel coordinates.

        Parameters
        ----------
        xyz_str : str
            Coordinate stored as a string.
        offset : list[int]
            Offset of coordinates in SWC file.

        Returns
        -------
        Tuple[int]
            xyz coordinates of an entry from an SWC file.

        """
        xyz = [float(xyz_str[i]) + offset[i] for i in range(3)]
        return img_util.to_voxels(xyz, self.anisotropy)


# --- Write ---
def to_zipped_point(zip_writer, filename, xyz):
    """
    Writes a point to an SWC file format, which is then stored in a ZIP
    archive.

    Parameters
    ----------
    zip_writer : zipfile.ZipFile
        A ZipFile object that will store the generated SWC file.
    filename : str
        Filename of SWC file.
    xyz : ArrayLike
        Point to be written to SWC file.

    Returns
    -------
    None

    """
    with StringIO() as text_buffer:
        # Preamble
        text_buffer.write("# COLOR 1.0 0.0 0.0")
        text_buffer.write("# id, type, z, y, x, r, pid")

        # Write entry
        x, y, z = tuple(xyz)
        text_buffer.write("\n" + f"1 2 {x} {y} {z} 10 -1")

        # Finish
        zip_writer.writestr(filename, text_buffer.getvalue())
