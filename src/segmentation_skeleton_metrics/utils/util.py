# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from io import BytesIO
from random import sample
from zipfile import ZipFile

import os
import shutil


# -- os utils ---
def mkdir(path, delete=False):
    """
    Creates a directory at "path".

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
    Removes directory and all subdirectories at "path".

    Parameters
    ----------
    path : str
        Path to directory and subdirectories to be deleted if they exist.

    Returns
    -------
    None

    """
    if os.path.exists(path):
        shutil.rmtree(path)


def list_dir(directory, extension=None):
    """
    Lists all filenames in "directory". If "extension" is provided, then only
    filenames ending with "extension" are returned.

    Parameters
    ----------
    directory : str
        Path to directory to be searched.

    extension : str, optional
       File type of interest. The default is None.

    Returns
    -------
    list[str]
        Filenames in the directory "directory".

    """
    if extension is None:
        return [f for f in os.listdir(directory)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(extension)]


def list_paths(directory, extension=None):
    """
    Lists all paths of files in the directory "directory". If "extension" is
    provided, then only paths of files ending with "extension" are returned.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    extension : str, optional
        File type of interest. The default is None.

    Returns
    -------
    list[str]
        Paths of files in the directory "directory".

    """
    paths = []
    for f in list_dir(directory, extension=extension):
        paths.append(os.path.join(directory, f))
    return paths


# --- io utils ---
def read_zip(zip_file, path):
    """
    Reads the content of an swc file from a zip file.

    Parameters
    ----------
    zip_file : ZipFile
        Zip containing text file to be read.

    Returns
    -------
    str
        Contents of a txt file.

    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


def read_txt(path):
    """
    Reads txt file stored at "path".

    Parameters
    ----------
    path : str
        Path where txt file is stored.

    Returns
    -------
    list[str]
        List where each entry corresponds to a line from the txt file.

    """
    with open(path, "r") as f:
        return f.read().splitlines()


def list_gcs_filenames(bucket, cloud_path, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    Parameters
    ----------
    bucket : google.cloud.client
        Client used to read from a GCS bucket.
    cloud_path : str
        ...
    extension : str
        File type of interest. The default is None.

    Returns
    -------
    list[str]
        Filenames in directory specified by "cloud_path" that end with
        "extension".

    """
    blobs = bucket.list_blobs(prefix=cloud_path)
    return [blob.name for blob in blobs if blob.name.endswith(extension)]


def list_files_in_zip(zip_content):
    """
    Lists all filenames in a zip.

    Parameters
    ----------
    zip_content : ...
        ...

    Returns
    -------
    list
        Filenames in a zip.

    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


# --- Miscellaneous ---
def get_segment_id(filename):
    return int(filename.split(".")[0])


def load_merged_labels(path):
    """
    Loads a list of merged label IDs from a text file.

    Parameters
    ----------
    path : str
        Path to text file containing the label IDs corresponding to known
        merge mistakes in a predicted segmentation.

    Returns
    -------
    list
        Integer IDs read from the text file.

    """
    merged_ids = list()
    for i, txt in enumerate(read_txt(path)):
        if i > 0:
            merged_ids.append(int(txt.split("-")[0]))
    return merged_ids


def load_valid_labels(path):
    """
    Loads the set of label IDs that are said to be 'valid', meaning that the
    corresponding fragments were not filtered out during the neuron
    reconstruction process. For example, this text file could contain the
    label IDs of all fragments with path length greater than 20ums.

    Parameters
    ----------
    path : str
        Path to txt file containing label IDs corresponding to fragments used
        in the neuron reconstruction process.

    Returns
    -------
    set
        Set of label IDs corresponding to fragments used in the neuron
        reconstruction process.

    """
    valid_labels = set()
    for label_str in read_txt(path):
        valid_labels.add(int(label_str.split(".")[0]))
    return valid_labels


def kdtree_query(kdtree, xyz):
    """
    Gets the xyz coordinates of the nearest neighbor of "xyz" from "kdtree".

    Parameters
    ----------
    xyz : tuple
        xyz coordinate to be queried.

    Returns
    -------
    tuple
        xyz coordinate of the nearest neighbor of "xyz".

    """
    _, idx = kdtree.query(xyz)
    return tuple(kdtree.data[idx])


def sample_once(my_container):
    """
    Samples a single element from "my_container".

    Parameters
    ----------
    my_container : container
        Container to be sampled from.

    Returns
    -------
    sample

    """
    return sample(my_container, 1)[0]


def time_writer(t, unit="seconds"):
    """
    Converts runtime "t" to its proper unit.

    Parameters
    ----------
    t : float
        Runtime to be converted.
    unit : str, optional
        Unit of "t". The default is "seconds".

    Returns
    -------
    float
        Converted runtime.
    str
        Unit of "t"

    """
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit
