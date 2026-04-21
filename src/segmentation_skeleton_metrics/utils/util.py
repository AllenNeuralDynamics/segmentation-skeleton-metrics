"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for helper routines.

"""

from botocore import UNSIGNED
from botocore.client import Config
from collections import deque
from random import sample
from google.cloud import storage
from io import BytesIO, StringIO
from zipfile import ZipFile

import boto3
import os
import pandas as pd
import shutil


# -- OS Utils ---
def mkdir(dir_path, delete=False):
    """
    Creates a directory at the given path.

    Parameters
    ----------
    dir_path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete the directory if it already exists
        Default is False.
    """
    if delete:
        rmdir(dir_path)

    os.makedirs(dir_path, exist_ok=True)


def rmdir(dir_path):
    """
    Removes the given directory and all of its subdirectories.

    Parameters
    ----------
    dir_path : str
        Path to directory to be removed if it exists.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def rm_file(path):
    """
    Removes the file at the given path.

    Parameters
    ----------
    path : str
        Path to file to be removed.
    """
    if os.path.exists(path):
        os.remove(path)


def list_dir(dir_path, extension=""):
    """
    Lists filenames in the given directory. If "extension" is provided,
    filenames ending with the given extension are returned.

    Parameters
    ----------
    dir_path : str
        Path to directory to be searched.
    extension : str, optional
       Extension of filenames to be returned. Default is an empty string.

    Returns
    -------
    List[str]
        Filenames in the given directory.
    """
    return [f for f in os.listdir(dir_path) if f.endswith(extension)]


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
    paths : List[str]
        Paths of files in the given directory.
    """
    paths = list()
    for f in list_dir(directory, extension=extension):
        paths.append(os.path.join(directory, f))
    return paths


# --- IO Utils ---
def read_json(path):
    """
    Reads JSON file located at the given path.

    Parameters
    ----------
    path : str
        Path to JSON file to be read.

    Returns
    -------
    dict
        Contents of JSON file.
    """
    return pd.read_json(path, storage_options={"anon": True}, typ="series")


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
    if is_s3_path(path):
        return read_txt_from_s3(path)
    elif is_gcs_path(path):
        return read_txt_from_gcs(path)
    else:
        with open(path, "r") as f:
            return f.read()


def read_zip(zip_file, path):
    """
    Reads txt file contained in the given ZIP archive.

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


def update_txt(path, text, verbose=True):
    """
    Appends the given text to a specified text file and prints the text.

    Parameters
    ----------
    path : str
        Path to txt file where the text will be appended.
    text : str
        Text to be written to the file.
    verbose : bool, optional
        Indication of whether to printout text. Default is True.
    """
    # Printout text (if applicable)
    if verbose:
        print(text)

    # Update txt file
    with open(path, "a") as file:
        file.write(text + "\n")


# --- Graph Utils ---
def compute_segmented_run_length(graph, results, name):
    """
    Computes the run length of a graph that was segmented.

    Parameters
    ----------
    graph : LabeledGraph
        Graph to be evaluated.
    results : pandas.DataFrame
        Data frame containing skeleton metrics

    Returns
    -------
    float
        Run length of a graph that was segmented.
    """
    omit_rl = graph.run_length * results["% Omit Edges"][name] / 100
    split_rl = graph.run_length * results["% Split Edges"][name] / 100
    return graph.run_length - omit_rl - split_rl


def search_branching_node(graph, kdtree, root, radius=40):
    """
    Searches for a branching node within distance "radius" from the given
    root node.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    kdtree : scipy.spatial.KDTree
        KDTree containing physical coordinates from a ground truth tracing.
    root : int
        Root of search.
    radius : float, optional
        Distance to search from root. Default is 40.

    Returns
    -------
    root : int
        Root node or closest branching node within distance "radius".
    """
    queue = deque([(root, 0)])
    visited = {root}
    while queue:
        # Visit node
        i, d_i = queue.popleft()
        xyz_i = graph.node_xyz(i)
        if graph.degree[i] > 2:
            dist, _ = kdtree.query(xyz_i)
            if dist < 16:
                return i

        # Update queue
        for j in graph.neighbors(i):
            d_j = d_i + graph.physical_dist(i, j)
            if j not in visited and d_j < radius:
                queue.append((j, d_j))
                visited.add(j)
    return root


# --- Cloud Utils ---
def parse_cloud_path(path):
    """
    Parses a cloud storage path into its bucket name and prefix. Supports
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
    # Split path
    path = path[len("s3://"):] if is_s3_path else path[len("gs://"):]
    parts = path.split("/", 1)

    # Extract bucket and prefix
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def list_cloud_paths(path, extension=""):
    """
    Lists all files in a GCS/S3 bucket with the given extension.

    Parameters
    ----------
    path : str
        Path to cloud prefix to be searched, must be in the format:
        f"{scheme}://{bucket_name}/{prefix}".
    extension : str, optional
        File extension of filenames to be listed. Default is an empty string.

    Returns
    -------
    List[str]
        Filenames stored at the GCS path with the given extension.
    """
    assert is_gcs_path(path) or is_s3_path(path)
    bucket_name, prefix = parse_cloud_path(path)
    list_fn = list_gcs_paths if is_gcs_path(path) else list_s3_paths
    return list_fn(bucket_name, prefix, extension=extension)


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


def list_gcs_paths(bucket_name, prefix, extension=""):
    """
    Lists paths at a GCS prefix with the given extension.

    Parameters
    ----------
    bucket_name : str
        Name of bucket containing prefix.
    prefix : str
        Path to location within bucket to be searched.
    extension : str, optional
        File extension of filenames to be listed. Default is an empty string.

    Returns
    -------
    List[str]
        Paths under the GCS prefix with the given extension.
    """
    bucket = storage.Client().bucket(bucket_name)
    paths = list()
    for name in [b.name for b in bucket.list_blobs(prefix=prefix)]:
        if extension in name:
            paths.append(os.path.join(f"gs://{bucket_name}", name))
    return paths


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
    subdirs : List[str]
         Direct subdirectories.
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


def read_txt_from_gcs(path):
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
    bucket_name, subpath = parse_cloud_path(path)
    bucket = storage.Client().bucket(bucket_name)
    return bucket.blob(subpath).download_as_text()


def upload_directory_to_gcs(bucket_name, src_dir, dst_dir):
    """
    Uploads the contents of a local directory to a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of bucket to be read from.
    src_dir : str
        Path to the local directory whose contents should be uploaded.
    dst_dir : str
        Prefix path in the GCS bucket under which the files will be stored.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(src_dir):
        for filename in files:
            local_path = os.path.join(root, filename)

            # Compute the relative path and GCS destination path
            path = os.path.relpath(local_path, start=src_dir)
            blob_path = os.path.join(dst_dir, path).replace("\\", "/")

            # Upload the file
            bucket.blob(blob_path).upload_from_filename(local_path)


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
        Prefix to search under.
    extension : str, optional
        File extension to filter by. Default is an empty string.

    Returns
    -------
    paths : List[str]
        S3 object keys that match the prefix and extension filter.
    """
    # Create an anonymous client for public buckets
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # List all objects under the prefix
    paths = list()
    if "Contents" in response:
        for obj in response["Contents"]:
            filename = obj["Key"]
            if filename.endswith(extension):
                path = os.path.join(f"s3://{bucket_name}", filename)
                paths.append(path)
    return paths


def read_txt_from_s3(path):
    """
    Reads a txt file stored in an S3 bucket.

    Parameters
    ----------
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    bucket_name, subpath = parse_cloud_path(path)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    obj = s3.get_object(Bucket=bucket_name, Key=subpath)
    return obj["Body"].read().decode("utf-8")


# --- Miscellaneous ---
def compute_weighted_avg(df, column_name):
    """
    Compute the weighted average of a specified column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the target column 'SWC Run Length'
        column used as weights.
    column_name : str
        Name of the column for which to compute the weighted average.

    Returns
    -------
    float
        Weighted average of the specified column, ignoring rows where either
        the value or weight is NaN. Returns NaN if the total weight is zero.
    """
    # Extract values
    values = df[column_name]
    weights = df["SWC Run Length"]

    # Ignore NaNs
    mask = values.notna() & weights.notna()
    values = values[mask]
    weights = weights[mask]

    # Compute weighted mean
    if weights.sum() == 0:
        return float("nan")
    else:
        return (values * weights).sum() / weights.sum()


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
    str
        Segment ID.
    """
    return filename.split(".")[0]


def sample_once(my_container):
    """
    Samples a single element from "my_container".

    Parameters
    ----------
    my_container : Container
        Container to be sampled from.

    Returns
    -------
    hashable
        Random element from the given container.
    """
    return sample(my_container, 1)[0]


def to_zipped_point(zip_file, filename, xyz):
    """
    Writes a point to an SWC file format, which is then stored in a ZIP
    archive.

    Parameters
    ----------
    zip_file : zipfile.ZipFile
        ZipFile object that writes the SWC file.
    filename : str
        Filename of SWC file.
    xyz : ArrayLike
        Point to be written to SWC file.
    """
    with StringIO() as text_buffer:
        # Preamble
        text_buffer.write("# COLOR 1.0 0.0 0.0")
        text_buffer.write("\n" + "# id, type, z, y, x, r, pid")

        # Write entry
        x, y, z = tuple(xyz)
        text_buffer.write("\n" + f"1 2 {x} {y} {z} 10 -1")

        # Finish
        zip_file.writestr(filename, text_buffer.getvalue())
