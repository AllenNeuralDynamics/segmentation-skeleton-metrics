"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for reading and processing images.

"""

from abc import ABC, abstractmethod
from cloudvolume import CloudVolume
from tifffile import imread

import io
import numpy as np
import os
import tensorstore as ts
import zipfile

from segmentation_skeleton_metrics.utils import util


class ImageReader(ABC):
    """
    Abstract base class for creating image reader subclasses.
    """

    def __init__(self, img_path):
        """
        Instantiates an ImageReader object.

        Parameters
        ----------
        img_path : str
            Path to an image.
        """
        self.img = None
        self.img_path = img_path
        self._load_image()

    @abstractmethod
    def _load_image(self):
        """
        This method should be implemented by subclasses to load the image
        based on the "img_path" attribute.
        """
        pass

    def read(self, voxel, shape):
        """
        Reads a patch from an image given a voxel coordinate and patch shape.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel coordinate of top-left-front corner of the image patch to be
            read.
        shape : Tuple[int]
            Shape of the image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        v1 = voxel
        v2 = [v1[i] + shape[i] for i in range(3)]
        if len(self.shape()) == 3:
            return self.img[v1[0]: v2[0], v1[1]: v2[1], v1[2]: v2[2]]
        elif len(self.shape()) == 4:
            return self.img[v1[0]: v2[0], v1[1]: v2[1], v1[2]: v2[2], 0]
        elif len(self.shape()) == 5:
            return self.img[0, 0, v1[0]: v2[0], v1[1]: v2[1], v1[2]: v2[2]]
        else:
            raise ValueError(f"Unsupported image shape: {self.shape()}")

    def read_with_bbox(self, bbox):
        """
        Reads the image patch defined by the given bounding box.

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
        return self.read(bbox["min"], shape)

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


class TensorStoreReader(ImageReader):
    """
    Class that reads an image with the TensorStore library.
    """

    def __init__(self, img_path):
        """
        Instantiates a TensorStore image reader.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        self.driver = self.get_driver(img_path)
        super().__init__(img_path)

    def get_driver(self, img_path):
        """
        Gets the storage driver needed to read the image.

        Returns
        -------
        str
            Storage driver needed to read the image.
        """
        if ".zarr" in img_path:
            return "zarr"
        elif ".n5" in img_path:
            return "n5"
        elif is_neuroglancer_precomputed(img_path):
            return "neuroglancer_precomputed"
        else:
            raise ValueError(f"Unsupported image format: {img_path}")

    def _load_image(self):
        """
        Loads image using the TensorStore library.
        """
        # Extract metadata
        bucket_name, path = util.parse_cloud_path(self.img_path)
        storage_driver = get_storage_driver(self.img_path)

        # Load image
        self.img = ts.open(
            {
                "driver": self.driver,
                "kvstore": {
                    "driver": storage_driver,
                    "bucket": bucket_name,
                    "path": path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": 1000000000},
                    "cache_pool#remote": {"total_bytes_limit": 1000000000},
                    "data_copy_concurrency": {"limit": 8},
                },
                "recheck_cached_data": "open",
            }
        ).result()

    def read(self, voxel, shape):
        """
        Reads a patch from an image given a voxel coordinate and patch shape.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel coordinate of the top-left-front corner of the image patch
            to be read.
        shape : Tuple[int]
            Shape of the image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        img_patch = super().read(voxel, shape)
        return img_patch.read().result()


class TiffReader(ImageReader):
    """
    Class that reads an image with the Tifffile library.
    """

    def __init__(self, img_path, inner_tiff=None, swap_axes=True):
        """
        Instantiates a TiffReader image reader.

        Parameters
        ----------
        img_path : str
            Path to a TIFF image or ZIP archive containing a TIFF image.
        inner_tiff : str or None, optional
            If img_path is a ZIP file, specifies the TIFF filename inside the
            ZIP. Default is None.
        swap_axes : bool, optional
            Indication of whether to swap axes 0 and 2. Default is True.
        """
        # Instance attributes
        self.inner_tiff = inner_tiff
        self.swap_axes = swap_axes

        # Call parent class
        super().__init__(img_path)

    def _load_image(self):
        """
        Loads image using the Tifffile library.
        """
        #  Read image
        if self.img_path.lower().endswith(".zip"):
            assert self.inner_tiff is not None, "Must provide TIFF filename!"
            self._load_zipped_image()
        else:
            self.img = imread(self.img_path)

        # Swap axes (if applicable)
        if self.swap_axes:
            self.img = np.swapaxes(self.img, 0, 2)

    def _load_zipped_image(self):
        """
        Loads an image in a ZIP archive using the Tifffile library.
        """
        with zipfile.ZipFile(self.img_path, "r") as z:
            # Collect only valid TIFF files, ignoring __MACOSX junk
            tiff_files = [
                f for f in z.namelist()
                if f.lower().endswith((".tif", ".tiff"))
                and not os.path.basename(f).startswith("._")
            ]

            # Choose file
            matches = [f for f in tiff_files if f.endswith(self.inner_tiff)]
            if not matches:
                raise FileNotFoundError(f"{self.inner_tiff} not found in ZIP")
            filename = matches[0]

            # Load TIFF
            with z.open(filename) as f:
                self.img = imread(io.BytesIO(f.read()))


# --- Helpers ---
def get_storage_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Image path to be checked.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if util.is_s3_path(img_path):
        return "s3"
    elif util.is_gcs_path(img_path):
        return "gcs"
    else:
        raise ValueError(f"Unsupported path type: {img_path}")


def is_neuroglancer_precomputed(path):
    """
    Checks if the path points to a neuroglancer precomputed dataset.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path points to a neuroglancer precomputed
        dataset.
    """
    try:
        vol = CloudVolume(path)
        return all(k in vol.info for k in ["data_type", "scales", "type"])
    except Exception:
        return False


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
