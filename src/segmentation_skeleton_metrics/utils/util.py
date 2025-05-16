# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Code for helper routines.

"""

from random import sample
from google.cloud import storage
from io import BytesIO
from xlwt import Workbook
from zipfile import ZipFile

import os
import shutil


# -- OS Utils ---
def mkdir(path, delete=False):
    """
    Creates a directory at the given path.

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
    Removes the given directory and all of its subdirectories.

    Parameters
    ----------
    path : str
        Path to directory to be removed if it exists.

    Returns
    -------
    None

    """
    if os.path.exists(path):
        shutil.rmtree(path)


def list_dir(directory, extension=None):
    """
    Lists filenames in the given directory. If "extension" is provided,
    filenames ending with the given extension are returned.

    Parameters
    ----------
    directory : str
        Path to directory to be searched.
    extension : str, optional
       Extension of filenames to be returned. The default is None.

    Returns
    -------
    List[str]
        Filenames in the given directory.

    """
    if extension is None:
        return [f for f in os.listdir(directory)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(extension)]


def list_paths(directory, extension=None):
    """
    Lists paths of files in the given directory. If "extension" is provided,
    filenames ending with the given extension are returned.

    Parameters
    ----------
    directory : str
        Path to directory to be searched.
    extension : str, optional
        Extension of filenames to be returned. The default is None.

    Returns
    -------
    list[str]
        Paths of files in the given directory.

    """
    paths = list()
    for f in list_dir(directory, extension=extension):
        paths.append(os.path.join(directory, f))
    return paths


# --- IO Utils ---
def read_zip(zip_file, path):
    """
    Reads a txt file contained in the given ZIP archive.

    Parameters
    ----------
    zip_file : ZipFile
        ZIP archive containing text file.

    Returns
    -------
    str
        Contents of a txt file.

    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


def read_txt(path):
    """
    Reads txt file at the given path.

    Parameters
    ----------
    path : str
        Path to txt file.

    Returns
    -------
    List[str]
        Lines from the txt file.

    """
    with open(path, "r") as f:
        return f.read().splitlines()


def update_txt(path, text):
    """
    Appends the given text to a specified text file and prints the text.

    Parameters
    ----------
    path : str
        Path to txt file where the text will be appended.
    text : str
        Text to be written to the file.

    Returns
    -------
    None

    """
    print(text)
    with open(path, 'a') as file:
        file.write(text + "\n")


# -- GCS utils --
def list_files_in_zip(zip_content):
    """
    Lists all files in a zip file stored in a GCS bucket.

    Parameters
    ----------
    zip_content : str
        Content stored in a zip file in the form of a string of bytes.

    Returns
    -------
    list[str]
        List of filenames in a zip file.

    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


def list_gcs_filenames(gcs_dict, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    Parameters
    ----------
    gcs_dict : dict
        ...
    extension : str
        File extension of filenames to be listed.

    Returns
    -------
    list
        Filenames stored at "cloud" path with the given extension.

    """
    bucket = storage.Client().bucket(gcs_dict["bucket_name"])
    blobs = bucket.list_blobs(prefix=gcs_dict["path"])
    return [blob.name for blob in blobs if extension in blob.name]


def list_gcs_subdirectories(bucket_name, prefix):
    """
    Lists all direct subdirectories of a given prefix in a GCS bucket.

    Parameters
    ----------
    bucket : str
        Name of bucket to be read from.
    prefix : str
        Path to directory in "bucket".

    Returns
    -------
    list[str]
         List of direct subdirectories.

    """
    # Load blobs
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter="/"
    )
    [blob.name for blob in blobs]

    # Parse directory contents
    prefix_depth = len(prefix.split("/"))
    subdirs = list()
    for prefix in blobs.prefixes:
        is_dir = prefix.endswith("/")
        is_direct_subdir = len(prefix.split("/")) - 1 == prefix_depth
        if is_dir and is_direct_subdir:
            subdirs.append(prefix)
    return subdirs


def upload_directory_to_gcs(bucket_name, source_dir, destination_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(source_dir):
        for filename in files:
            local_path = os.path.join(root, filename)

            # Compute the relative path and GCS destination path
            relative_path = os.path.relpath(local_path, start=source_dir)
            blob_path = os.path.join(destination_dir, relative_path).replace("\\", "/")

            # Upload the file
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)


# --- Miscellaneous ---
def get_segment_id(filename):
    """
    Gets the segment ID correspionding to the given filename, assuming that
    the format of filename is "{segment_id}.{*anything*}.swc".

    Parameters
    ----------
    filename : str
        Name of file to extract segmentation ID from.

    Returns
    -------
    int
        Segment ID.

    """
    return int(filename.split(".")[0])


def load_merged_labels(path):
    """
    Loads segment IDs that are known to contain a merge mistake from the
    given txt file.

    Parameters
    ----------
    path : str
        Path to txt file containing segment IDs.

    Returns
    -------
    List[int]
        Segment IDs that are known to contain a merge mistake.

    """
    merged_ids = list()
    for i, txt in enumerate(read_txt(path)):
        if i > 0:
            merged_ids.append(int(txt.split("-")[0]))
    return merged_ids


def load_valid_labels(path):
    """
    Loads segment IDs that can be assigned to nodes, accounts for segments
    that may have been removed due to some type of filtering. The default is
    None.

    Parameters
    ----------
    path : str
        Path to txt file containing segment IDs.

    Returns
    -------
    Set[int]
        Segment IDs that can be assigned to nodes.

    """
    valid_labels = set()
    for label_str in read_txt(path):
        valid_labels.add(int(label_str.split(".")[0]))
    return valid_labels


def kdtree_query(kdtree, xyz):
    """
    Gets the nearest neighbor of the given xyz coordinate from "kdtree".

    Parameters
    ----------
    xyz : ArrayLike
        Coordinate to be queried.

    Returns
    -------
    Tuple[float]
        Coordinate of the nearest neighbor in the given KD-Tree.

    """
    _, idx = kdtree.query(xyz)
    return tuple(kdtree.data[idx])


def sample_once(my_container):
    """
    Samples a single element from "my_container".

    Parameters
    ----------
    my_container : Container
        Container to be sampled from.

    Returns
    -------
    Hashable
        Random element from the given container.

    """
    return sample(my_container, 1)[0]


def save_results(path, stats):
    """
    Saves the evaluation results generated from skeleton-based metrics to an
    Excel file.

    Parameters
    ----------
    path : str
        Path where the Excel file will be saved.
    stats : dict
        Dictionary where the keys are SWC IDs (as strings) and the values
        are dictionaries containing metrics as keys and their respective
        values.

    Returns
    -------
    None

    """
    # Initialize
    wb = Workbook()
    sheet = wb.add_sheet("Results")
    sheet.write(0, 0, "swc_id")

    # Label rows and columns
    swc_ids = list(stats.keys())
    for i, swc_id in enumerate(swc_ids):
        sheet.write(i + 1, 0, swc_id)

    metrics = list(stats[swc_id].keys())
    for i, metric in enumerate(metrics):
        sheet.write(0, i + 1, metric)

    # Write stats
    for i, swc_id in enumerate(swc_ids):
        for j, metric in enumerate(metrics):
            sheet.write(i + 1, j + 1, round(stats[swc_id][metric], 4))

    wb.save(path)
