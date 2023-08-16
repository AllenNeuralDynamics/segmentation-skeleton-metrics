# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from abc import ABC, abstractmethod

import numpy as np
from tifffile import imwrite

from aind_segmentation_evaluation import nx_utils, swc_utils, utils

SUPPORTED_FILETYPES = ["tif", "n5"]


class SegmentationMetrics(ABC):
    """
    Class that evaluates a segmentation in terms of the number of
    splits and merges.

    """

    def __init__(
        self,
        swc_dir,
        labels,
        anisotropy=[1.0, 1.0, 1.0],
        filetype=None,
        log_dir=None,
        img_log=False,
        swc_log=False,
        txt_log=False,
    ):
        """
        Constructs object that evaluates a segmentation mask.

        Parameters
        ----------
        graph : list[networkx.Graph]
            List of graphs where each graph represents a neuron.
        labels : dict
            Segmentation mask.
        ...

        Returns
        -------
        None.

        """
        # Graph and labels
        self.graphs = swc_utils.dir_to_graphs(swc_dir, anisotropy=anisotropy)
        if type(labels) is str:
            self.labels = self.init_labels(pred_labels, filetype)
        else:
            self.labels = labels

        # Mistake trackers
        self.site_cnt = 0
        self.edge_cnt = 0
        if type(labels) == dict:
            self.site_mask = dict()
        else:
            self.site_mask = np.zeros(labels.shape, dtype=bool)

        # Mistake logs
        self.log_dir = log_dir
        self.img_log = img_log
        self.swc_log = swc_log
        self.txt_log = txt_log
        if self.log_dir is not None:
            utils.rmdir(self.log_dir)
            utils.mkdir(self.log_dir)
        if self.img_log:
            assert type(labels) == np.array
            self.edge_mask = np.zeros(labels.shape, dtype=bool)
        if self.swc_log:
            self.swc_dir = os.path.join(self.log_dir, "swc_files")
            utils.mkdir(self.swc_dir)
        if self.txt_log:
            self.mistakes_log = ["# xyz1,  xyz2,  swc1,  swc2"]

    def init_labels(self, path, filetype):
        """
        Initializes a volume by uploading file with extension "filetype".

        Parameters
        ----------
        path : str
            Path to image volume.
        filetype : str
            Extension of file to be uploaded, supported values include tif, n5,
            and tensorstore.

        Returns
        -------
        dict
            Sparse image volume of segmentation mask.

        """
        assert (
            filetype is not None
        ), "Must provide filetype to upload image volumes!"
        assert filetype in SUPPORTED_FILETYPES, "Filetype is not supported!"
        if filetype == "tensorstore":
            return utils.read_tensorstore(path)
        elif filetype == "n5":
            return utils.read_n5(path)
        else:
            return utils.read_tif(path)

    def get_labels(self):
        if type(self.labels) == np.array:
            labels = np.unique(self.labels)
        else:
            labels = np.unique(list(self.labels.keys()))
        return [label for label in labels if label != 0]

    def count_edges(self):
        """
        Counts number of edges in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        total_edges = 0
        for graph in self.graphs:
            total_edges += graph.number_of_edges()
        return total_edges

    def is_mistake(self, a, b):
        """
        Checks if "a" and "b" are positive and not equal.

        Parameters
        ----------
        a : int
            label at node i.
        b : int
            label at node j.

        Returns
        -------
        bool
            Indicates whether there is a mistake.

        """
        return (a != 0 and b != 0) and (a != b)


    def log_site(self, graph, edge, prefix):
        """
        Logs xyz coordinate of mistake in a numpy.array
        or writes an swc file.

        Parameters
        ----------
        graph : networkx.Graph
            Graph that represents a neuron.
        node_tuple : tuple
            Node of "graph".
        fn : str
            Filename of swc that will be written.

        Returns
        -------
        None

        """
        # Get site info
        xyz_i, xyz_j = nx_utils.get_edge_xyz(graph, edge)

        # Log img info
        self.site_mask[xyz_i] = 1
        self.site_mask[xyz_j] = 1
        if self.img_log:
            self.edge_mask[xyz_i] = 1
            self.edge_mask[xyz_j] = 1

        # Log swc info
        if self.swc_log:
            red = " 1.0 0.0 0.0"
            list_of_entries = [swc_utils.make_entry(xyz_i, 10, -1)]
            fn = prefix + str(self.site_cnt) + ".swc"
            path = os.path.join(self.swc_dir, fn)
            swc_utils.write_swc(path, list_of_entries, color=red)

        # Log txt info
        if self.txt_log:
            label_i, label_j = nx_utils.get_labels(self.labels, graph, edge)
            xyz = str(xyz_i) + ", " + str(xyz_j) + ", "
            labels = str(label_i) + ", " + str(label_j)
            self.mistakes_log.append(xyz + labels)

    def write_results(self, fn):
        """
        Writes "site_mask" and "edge" mask to.

        Parameters
        ----------
        fn : str
            Filename.

        Returns
        -------
        None.

        """
        # Write img log
        if self.img_log:
            site_path = os.path.join(self.log_dir, fn + "_sites.tif")
            edges_path = os.path.join(self.log_dir, fn + "_edges.tif")
            imwrite(site_path, self.site_mask)
            imwrite(edges_path, self.edge_mask)

        # Write txt log
        if self.txt_log:
            path = os.path.join(self.log_dir, fn + ".txt")
            utils.write_txt(path, self.mistakes_log)


    @abstractmethod
    def detect_mistakes(self):
        """
        Detects differences between corresponding labels of graph and volume.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def mistake_search(self):
        """
        Determines whether a complex mistake is a misalignment between the
        volume and graph or a true mistake.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pass
