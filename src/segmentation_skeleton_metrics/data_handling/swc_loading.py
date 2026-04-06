"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for working with SWC files. An SWC file is a text-based file format
used to represent the directed graphical structure of a neuron. It contains a
series of nodes such that each has the following attributes:
    "id" (int): node ID
    "type" (int): node type (e.g. soma, axon, dendrite)
    "x" (float): x coordinate
    "y" (float): y coordinate
    "z" (float): z coordinate
    "pid" (int): node ID of parent

Note: Each line in an SWC file corresponds to a node and contains these
      attributes in the same order.
"""

from botocore import UNSIGNED
from botocore.client import Config
from collections import deque
from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from google.auth.exceptions import RefreshError
from google.cloud import storage
from io import BytesIO
from tqdm import tqdm
from zipfile import ZipFile

import boto3
import numpy as np
import os

from segmentation_skeleton_metrics.utils import util


class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, and (3) local directory of ZIP archives.
    """

    def __init__(self, selected_ids=None, verbose=True):
        """
        Initializes a Reader object that reads SWC files.

        Parameters
        ----------
        selected_ids : Set[int], optional
            Only SWC files with an swc_id contained in this set are read.
            Default is None.
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        self.selected_ids = selected_ids or set()
        self.verbose = verbose

    # --- Read Data ---
    def __call__(self, swc_pointer):
        """
        Loads SWC files based on the type pointer provided.

        Parameters
        ----------
        swc_pointer : str
            Object that points to SWC files to be read, must be one of:
                - file_path: Path to single SWC file
                - dir_path: Path to local directory with SWC files
                - zip_path: Path to local ZIP with SWC files
                - zip_dir_path: Path to local directory of ZIPs with SWC files
                - s3_dir_path: Path to S3 prefix with SWC files
                - gcs_dir_path: Path to GCS prefix with SWC files
                - gcs_zip_dir_path: Path to GCS prefix with ZIPs of SWC files

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from the SWC files. Each dictionary contains the following:
            items:
                - "id": unique identifier of each node in an SWC file.
                - "pid": parent ID of each node.
                - "radius": radius value corresponding to each node.
                - "xyz": coordinate corresponding to each node.
                - "filename": filename of SWC file
                - "swc_id": name of SWC file, minus the ".swc".
        """
        # Directory containing...
        if os.path.isdir(swc_pointer):
            # Local ZIP archives with SWC files
            paths = util.list_paths(swc_pointer, extension=".zip")
            if len(paths) > 0:
                return self.read_zips(swc_pointer, self.read_zip)

            # Local SWC files
            paths = util.read_paths(swc_pointer, extension=".swc")
            if len(paths) > 0:
                return self.read_swcs(paths, self.read_swc)

            raise Exception("Directory is Invalid!")

        # Path to...
        if isinstance(swc_pointer, str):
            # Cloud GCS/S3 storage
            if util.is_gcs_path(swc_pointer) or util.is_s3_path(swc_pointer):
                return self.read_from_cloud(swc_pointer)

            # Local ZIP archive with SWC files
            if swc_pointer.endswith(".zip"):
                return self.read_zip(swc_pointer)

            # Local path to single SWC file
            if swc_pointer.endswith(".swc"):
                return self.read_swc(swc_pointer)

            raise Exception("Path is Invalid!")

        raise Exception("SWC Pointer is Invalid!")

    def read_swc(self, path):
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

    def read_swcs(self, swc_paths, read_fn):
        """
        Reads SWC files stored in a GCS or S3 bucket.

        Parameters
        ----------
        bucket_name : str
            Name of bucket containing SWC files.
        swc_paths : List[str]
            List of paths to SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = set()
            for path in swc_paths:
                if self.confirm_read(os.path.basename(path)):
                    threads.add(executor.submit(read_fn, path))
                break

            # Store results
            swc_dicts = deque()
            pbar = self.manual_progress_bar(len(threads))
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
                if self.verbose:
                    pbar.update(1)
        return swc_dicts

    def read_zips(self, zip_paths, read_fn):
        """
        Reads SWC files stored in ZIP archives.

        Parameters
        ----------
        bucket_name : str
            Name of bucket containing SWC files.
        zip_paths : List[str]
            Paths to ZIP archives containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        pbar = tqdm(total=len(zip_paths), desc="Read SWCs")
        with ProcessPoolExecutor() as executor:
            # Assign processes
            futures = {executor.submit(read_fn, path) for path in zip_paths}

            # Store results
            swc_dicts = deque()
            for process in as_completed(futures):
                try:
                    swc_dicts.extend(process.result())
                except RefreshError:
                    pass

                if self.verbose:
                    pbar.update(1)
        return swc_dicts

    def read_zip(self, zip_path):
        """
        Reads SWC files from a ZIP archive.

        Paramters
        ---------
        zip_path : str
            Path to ZIP archive.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            zf = ZipFile(zip_path, "r")
            for name in [f for f in zf.namelist() if f.endswith(".swc")]:
                if self.confirm_read(name):
                    threads.append(
                        executor.submit(self.read_zipped_swc, zf, name
                        )
                    )

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    def read_zipped_swc(self, zipfile, path):
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
        return self.parse(content, filename)

    def read_from_cloud(self, path):
        """
        Reads SWC files stored in a GCS or S3 bucket.

        Parameters
        ----------
        path : str
            Path to location in a GCS or S3 bucket containing SWC files,
            must be in the format "{scheme}://{bucket_name}/{prefix}".

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Extact info
        assert util.is_s3_path(path) or util.is_gcs_path(path)
        use_s3 = util.is_s3_path(path)

        # List filenames
        swc_paths = util.list_cloud_paths(path, ".swc")
        zip_paths = util.list_cloud_paths(path, ".zip")

        # Call reader
        if swc_paths:
            read_fn = self.read_s3_swc if use_s3 else self.read_gcs_swc
            return self.read_swcs(swc_paths, read_fn)
        elif zip_paths:
            read_fn = self.read_s3_zip if use_s3 else self.read_gcs_zip
            return self.read_zips(zip_paths, read_fn)

        raise Exception(f"SWC Pointer is invalid {path}")

    def read_gcs_swc(self, path):
        """
        Reads SWC files stored in ZIP archives stored in a GCS or S3 bucket.

        Parameters
        ----------
        path : List[str]
            Path to SWC file to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Initialize cloud reader
        bucket_name, subpath = util.parse_cloud_path(path)
        bucket = storage.Client().bucket(bucket_name)
        blob = bucket.blob(subpath)

        # Parse swc contents
        content = blob.download_as_text().splitlines()
        filename = os.path.basename(subpath)
        return self.parse(content, filename)

    def read_gcs_zip(self, path):
        """
        Reads SWC files stored in a ZIP archive downloaded from a GCS
        bucket.

        Parameters
        ----------
        path : str
            Path to ZIP archive containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Initialize cloud reader
        client = storage.Client()
        bucket_name, path = util.parse_cloud_path(path)
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
                                self.read_zipped_swc, zip_file, filename
                            )
                        )

                # Process results
                for thread in as_completed(threads):
                    result = thread.result()
                    if result:
                        swc_dicts.append(result)
        return swc_dicts

    def read_s3_zip(self, path):
        """
        Reads SWC files stored in a ZIP archive downloaded from an S3
        bucket.

        Parameters
        ----------
        path : str
            Path to ZIP archive containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Initialize cloud reader
        bucket, key = util.parse_cloud_path(path)
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        zip_content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

        # Parse ZIP
        with ZipFile(BytesIO(zip_content), "r") as zf:
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = set()
                for name in zf.namelist():
                    if self.confirm_read(name):
                        threads.add(
                            executor.submit(self.read_zipped_swc, zf, filename
                        )

                # Store results
                swc_dicts = deque()
                for thread in as_completed(threads):
                    result = thread.result()
                    if result:
                        swc_dicts.append(result)
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
    def iterator(self, iterator):
        """
        Gets an iterator that optionally displays a progress bar.

        Parameters
        ----------
        iterator : iterable
            Object to be iterated over.

        Returns
        -------
        tqdm.tqdm
            Iterator that is optionally wrapped in a progress bar.
        """
        return tqdm(iterator, desc="Read SWCs") if self.verbose else iterator

    def manual_progress_bar(self, total):
        """
        Gets progress bar that needs to be updated manually.

        Parameters
        ----------
        total : int
            Size of progress bar.

        Returns
        -------
        tqdm.tqdm
            Iterator that is optionally wrapped in a progress bar.
        """
        return tqdm(total=total, desc="Read SWCs") if self.verbose else None

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
        swc_name, _ = os.path.splitext(filename)
        content, offset = self.process_content(content)
        if len(content) > 30:
            swc_dict = {
                "id": np.zeros((len(content)), dtype=int),
                "pid": np.zeros((len(content)), dtype=int),
                "voxel": np.zeros((len(content), 3), dtype=np.int32),
                "swc_name": swc_name,
            }

            # Parse content
            for i, line in enumerate(content):
                parts = line.split()
                swc_dict["id"][i] = parts[0]
                swc_dict["pid"][i] = parts[-1]
                swc_dict["voxel"][i] = self.read_coordinate(parts[2:5], offset)
            return swc_dict
        else:
            return None

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
        content : List[str]
            Lines from an SWC file after comments.
        offset : Tuple[int]
            Offset used to shift coordinate.
        """
        offset = (0, 0, 0)
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = self.read_coordinate(parts[2:5])
            if not line.startswith("#") and len(line.strip()) > 0:
                return content[i:], offset

    def read_coordinate(self, xyz_str, offset=(0, 0, 0)):
        """
        Reads a coordinate from a string and converts it to voxel coordinates.

        Parameters
        ----------
        xyz_str : str
            Coordinate stored as a string.
        offset : Tuple[int]
            Offset of coordinates in SWC file. Default is (0, 0, 0).

        Returns
        -------
        Tuple[int]
            xyz coordinates of an entry from an SWC file.
        """
        return [float(xyz_str[i]) + offset[i] for i in range(3)]
