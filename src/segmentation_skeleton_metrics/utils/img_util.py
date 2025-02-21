"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Code for reading images.

"""

from abc import ABC, abstractmethod
from tifffile import imread

import tensorstore as ts


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
        based on the "img_path" attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None

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
        return self.read(bbox["min"], shape)

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
    Class that reads an image with the TensorStore library.

    """

    def __init__(self, img_path, driver):
        """
        Instantiates a TensorStore image reader.

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

    def __init__(self, img_path):
        """
        Instantiates a TiffReader image reader.

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
        Loads image using the Tifffile library.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.img = imread(self.img_path)


# --- Miscellaneous ---
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
