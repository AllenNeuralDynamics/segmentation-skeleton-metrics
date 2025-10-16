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

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import pandas as pd
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
        exists. Default is False.
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
       Extension of filenames to be returned. Default is None.

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
        Extension of filenames to be returned. Default is None.

    Returns
    -------
    List[str]
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
        ZIP archive containing TXT file.

    Returns
    -------
    str
        Contents of a TXT file.
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
    """
    print(text)
    with open(path, "a") as file:
        file.write(text + "\n")


# -- GCS Utils --
def is_gcs_path(path):
    """
    Checks if the path is a GCS path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is a GCS path.
    """
    return path.startswith("gs://")


def list_files_in_zip(zip_content):
    """
    Lists all files in a zip file stored in a GCS bucket.

    Parameters
    ----------
    zip_content : str
        Content stored in a ZIP archive in the form of a string of bytes.

    Returns
    -------
    List[str]
        Filenames in a ZIP archive file.
    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


def list_gcs_filenames(bucket_name, prefix, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    Parameters
    ----------
    bucket_name : str
        Name of bucket to be searched.
    prefix : str
        Path to location within bucket to be searched.
    extension : str
        File extension of filenames to be listed.

    Returns
    -------
    List[str]
        Filenames stored at the GCS path with the given extension.
    """
    bucket = storage.Client().bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
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
    List[str]
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


def read_txt_from_gcs(bucket_name, path):
    """
    Reads a txt file stored in a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of bucket to be read from.
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.download_as_text()


def upload_directory_to_gcs(bucket_name, source_dir, destination_dir):
    """
    Uploads the contents of a local directory to a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of bucket to be read from.
    source_dir : str
        Path to the local directory whose contents should be uploaded.
    destination_dir : str
        Prefix path in the GCS bucket under which the files will be stored.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(source_dir):
        for filename in files:
            local_path = os.path.join(root, filename)

            # Compute the relative path and GCS destination path
            path = os.path.relpath(local_path, start=source_dir)
            blob_path = os.path.join(destination_dir, path).replace("\\", "/")

            # Upload the file
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)


# --- S3 Utils ---
def is_s3_path(path):
    """
    Checks if the given path is an S3 path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is an S3 path.
    """
    return path.startswith("s3://")


def list_s3_paths(bucket_name, prefix, extension=""):
    """
    Lists all object keys in a public S3 bucket under a given prefix,
    optionally filters by file extension.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    prefix : str
        The S3 "directory" prefix to search under.
    extension : str, optional
        File extension to filter by. Default is an empty string, which returns
        all files.

    Returns
    -------
    List[str]
        List of S3 object keys that match the prefix and extension filter.
    """
    # Create an anonymous client for public buckets
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # List all objects under the prefix
    filenames = list()
    if "Contents" in response:
        for obj in response["Contents"]:
            filename = obj["Key"]
            if filename.endswith(extension):
                filenames.append(filename)
    return filenames


def read_txt_from_s3(bucket_name, path):
    """
    Reads a txt file stored in an S3 bucket.

    Parameters
    ----------
    bucket_name : str
        Name of bucket to be read from.
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    obj = s3.get_object(Bucket=bucket_name, Key=path)
    return obj['Body'].read().decode('utf-8')


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
    df = pd.read_csv(path)
    return list(df["Segment_ID"])


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
    kdtree : scipy.spatial.KDTree
        KD-Tree to be searched.
    xyz : ArrayLike
        Coordinate to be queried.

    Returns
    -------
    Tuple[float]
        Coordinate of the nearest neighbor in the given KD-Tree.
    """
    _, idx = kdtree.query(xyz)
    return tuple(kdtree.data[idx])


def parse_cloud_path(path):
    """
    Parses a cloud storage path into its bucket name and key/prefix. Supports
    paths of the form: "{scheme}://bucket_name/prefix" or without a scheme.

    Parameters
    ----------
    path : str
        Path to be parsed.

    Returns
    -------
    bucket_name : str
        Name of the bucket.
    prefix : str
        Cloud prefix.
    """
    # Remove s3:// if present
    if is_s3_path(path):
        path = path[len("s3://"):]

    # Remove gs:// if present
    if is_gcs_path(path):
        path = path[len("gs://"):]

    # Extract bucket and prefix
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


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

