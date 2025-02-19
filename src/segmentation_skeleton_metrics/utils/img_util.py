"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Helper routines for reading and processing images.

"""

from abc import ABC, abstractmethod
from tifffile import imread

import numpy as np
import tensorstore as ts

from segmentation_skeleton_metrics.utils import util


class ImageReader(ABC):
    """
    Abstract class to create image readers classes.

    """

    def __init__(self, img_path):
        """
        Class constructor of image reader.

        Parameters
        ----------
        img_path : str
            Path to image.

        Returns
        -------
        None

        """
        self.img = None
        self.img_path = img_path
        self._load_image()

    @abstractmethod
    def _load_image(self):
        """
        This method should be implemented by subclasses to load the image
        based on img_path.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pass

    def read(self, voxel, shape, from_center=True):
        """
        Reads a patch from an image given a voxel coordinate and patch shape.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel coordinate that is either the center or top-left-front
            corner of the image patch to be read.
        shape : Tuple[int]
            Shape of the image patch to be read.
        from_center : bool, optional
            Indication of whether "voxel" is the center or top-left-front
            corner of the image patch to be read. The default is True.

        Returns
        -------
        numpy.ndarray
            Image patch.

        """
        s, e = get_start_end(voxel, shape, from_center=from_center)
        if len(self.shape()) == 3:
            return self.img[s[0]: e[0], s[1]: e[1], s[2]: e[2]]
        elif len(self.shape()) == 4:
            return self.img[s[0]: e[0], s[1]: e[1], s[2]: e[2], 0]
        elif len(self.shape()) == 5:
            return self.img[0, 0, s[0]: e[0], s[1]: e[1], s[2]: e[2]]
        else:
            raise ValueError(f"Unsupported image shape: {self.shape()}")

    def read_with_bbox(self, bbox):
        """
        Reads the image patch defined by a given bounding box.

        Parameters
        ----------
        bbox : dict
            Dictionary that contains min and max coordinates of a bounding
            box.

        Returns
        -------
        numpy.ndarray
            Image patch.

        """
        shape = [bbox["max"][i] - bbox["min"][i] for i in range(3)]
        return self.read(bbox["min"], shape, from_center=False)

    def read_voxel(self, voxel):
        """
        Reads the intensity value at a given voxel.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel to be read.

        Returns
        -------
        int
            Intensity value at voxel.

        """
        return self.img[voxel]

    def shape(self):
        """
        Gets the shape of image.

        Parameters
        ----------
        None

        Returns
        -------
        Tuple[int]
            Shape of image.

        """
        return self.img.shape


class TensorStoreReader(ImageReader):
    """
    Class that reads an image with TensorStore library.

    """

    def __init__(self, img_path, driver):
        """
        Constructs a TensorStore image reader.

        Parameters
        ----------
        img_path : str
            Path to image.
        driver : str
            Storage driver needed to read the image.

        Returns
        -------
        None

        """
        self.driver = driver
        super().__init__(img_path)

    def _load_image(self):
        """
        Loads image using the TensorStore library.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.img = ts.open(
            {
                "driver": self.driver,
                "kvstore": {
                    "driver": "gcs",
                    "bucket": "allen-nd-goog",
                    "path": self.img_path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": 1000000000},
                    "cache_pool#remote": {"total_bytes_limit": 1000000000},
                    "data_copy_concurrency": {"limit": 8},
                },
                "recheck_cached_data": "open",
            }
        ).result()
        if self.driver == "neuroglancer_precomputed":
            return self.img[ts.d["channel"][0]]

    def read(self, voxel, shape, from_center=True):
        """
        Reads a patch from an image given a voxel coordinate and patch shape.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel coordinate that is either the center or top-left-front
            corner of the image patch to be read.
        shape : Tuple[int]
            Shape of the image patch to be read.
        from_center : bool, optional
            Indication of whether "voxel" is the center or top-left-front
            corner of the image patch to be read. The default is True.

        Returns
        -------
        numpy.ndarray
            Image patch.

        """
        img_patch = super().read(voxel, shape, from_center)
        return img_patch.read().result()

    def read_voxel(self, voxel):
        """
        Reads the intensity value at a given voxel.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel to be read.

        Returns
        -------
        int
            Intensity value at voxel.

        """
        return int(self.img[voxel].read().result())


class TiffReader(ImageReader):
    """
    Class that reads image with Zarr library.

    """

    def __init__(self, img_path):
        """
        Constructs a Zarr image reader.

        Parameters
        ----------
        img_path : str
            Path to image.

        Returns
        -------
        None

        """
        super().__init__(img_path)

    def _load_image(self):
        """
        Loads image using the Zarr library.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.img = imread(self.img_path)


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    voxel : tuple
        Voxel coordinate that is either the center or top-left-front corner of
        the image patch to be read.
    shape : Tuple[int]
        Shape of the image patch to be read.
    from_center : bool, optional
        Indication of whether "voxel" is the center or top-left-front corner
        of the image patch to be read. The default is True.

    Return
    ------
    Tuple[List[int]]
        Start and end indices of the image patch to be read.

    """
    if from_center:
        start = [voxel[i] - shape[i] // 2 for i in range(3)]
        end = [voxel[i] + shape[i] // 2 for i in range(3)]
    else:
        start = voxel
        end = [voxel[i] + shape[i] for i in range(3)]
    return start, end


# --- Coordinate Conversions ---
def to_physical(voxel, anisotropy):
    """
    Converts a voxel coordinate to a physical coordinate by applying the
    anisotropy scaling factors.

    Parameters
    ----------
    voxel : ArrayLike
        Voxel coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.

    Returns
    -------
    Tuple[float]
        Physical coordinate.

    """
    return tuple([voxel[i] * anisotropy[i] for i in range(3)])


def to_voxels(xyz, anisotropy):
    """
    Converts coordinate from a physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordiante to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.

    Returns
    -------
    Tuple[int]
        Voxel coordinate.

    """
    return tuple([int(xyz[i] / anisotropy[i]) for i in range(3)])


# --- miscellaneous ---
def find_img_path(bucket_name, root_dir, dataset_name):
    """
    Finds the path to an image in a GCS bucket for the dataset given by
    "dataset_name".

    Parameters:
    ----------
    bucket_name : str
        Name of the GCS bucket where the images are stored.
    root_dir : str
        Path to the directory in the GCS bucket where the image is expected to
        be located.
    dataset_name : str
        Name of the dataset to be searched for within the subdirectories.

    Returns:
    -------
    str
        Path of the found dataset subdirectory within the specified GCS bucket.

    """
    for subdir in util.list_gcs_subdirectories(bucket_name, root_dir):
        if dataset_name in subdir:
            return subdir + "whole-brain/fused.zarr/"
    raise f"Dataset not found in {bucket_name} - {root_dir}"


def get_minimal_bbox(voxels, buffer=0):
    """
    Gets the min and max coordinates of a bounding box that contains "voxels".

    Parameters
    ----------
    voxels : numpy.ndarray
        Array containing voxel coordinates.
    buffer : int, optional
        Constant value added/subtracted from the max/min coordinates of the
        bounding box. The default is 0.

    Returns
    -------
    dict
        Bounding box.

    """
    bbox = {
        "min": np.floor(np.min(voxels, axis=0) - buffer).astype(int),
        "max": np.ceil(np.max(voxels, axis=0) + buffer).astype(int),
    }
    return bbox


def is_contained(bbox, voxel):
    """
    Checks whether a given voxel is contained within the image bounding box
    specified by "bbox".

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxel : Tuple[int]
        Voxel coordinate to be checked.

    Returns
    -------
    bool
        Inidcation of whether "voxel" is contained within the given image
        bounding box.

    """
    above = any([v >= bbox_max for v, bbox_max in zip(voxel, bbox["max"])])
    below = any([v < bbox_min for v, bbox_min in zip(voxel, bbox["min"])])
    return False if above or below else True


def is_list_contained(bbox, voxels):
    """
    Checks whether a list of voxels is contained within a given image bounding
    box.

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxels : List[Tuple[int]]
        List of voxel coordinates to be checked.

    Returns
    -------
    bool
        Indication of whether every element in "voxels" is contained in
        "bbox".

    """
    return all([is_contained(bbox, voxel) for voxel in voxels])
