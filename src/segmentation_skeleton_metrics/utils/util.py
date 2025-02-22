# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Code for helper routines.

"""

from io import BytesIO
from random import sample
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
