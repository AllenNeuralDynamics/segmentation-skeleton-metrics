# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from abc import ABC, abstractmethod

import numpy as np
import tensorstore as ts

from segmentation_skeleton_metrics import nx_utils, swc_utils, utils

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
        prefix="",
        log_dir=None,
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
        self.anisotropy = anisotropy
        self.graphs = swc_utils.dir_to_graphs(swc_dir, anisotropy=anisotropy)
        if type(labels) is str:
            self.labels = self.init_labels(labels, filetype)
        else:
            self.labels = labels

        # Mistake trackers
        self.site_cnt = 0
        self.edge_cnt = 0

        # Mistake logs
        self.prefix = prefix
        self.log_dir = log_dir
        self.swc_log = swc_log
        self.txt_log = txt_log
        if self.log_dir is not None:
            utils.mkdir(self.log_dir)
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
            Image volume.

        """
        assert filetype is not None, "Must provide filetype to upload image!"
        assert filetype in SUPPORTED_FILETYPES, "Filetype is not supported!"
        if filetype == "tensorstore":
            return utils.read_tensorstore(path)
        elif filetype == "n5":
            return utils.read_n5(path)
        else:
            return utils.read_tif(path)

    def get_labels(self):
        """
        Gets list of all unique labels in "self.labels".

        Parameters
        ----------
        None

        Returns
        -------
        list[int]
            List of all unique labels in "self.labels".

        """
        if type(self.labels) == np.array:
            labels = np.unique(self.labels)
        else:
            labels = np.unique(list(self.labels.keys()))
        return [label for label in labels if label != 0]

    def get_label(self, graph, i):
        """
        Gets label of node "i".

        Parameters
        ----------
        graph : networkx.Graph
            Graph which represents a neuron.
        i : int
            Node in "graph".

        Returns
        -------
        int
           Label of node "i".

        """
        return self._get_label(nx_utils.get_xyz(graph, i))

    def _get_label(self, xyz):
        """
        Gets label at image coordinates "xyz".

        Parameters
        ----------
        xyz : tuple[int]
            Coordinates that index into "self.labels".

        Returns
        -------
        int
           Label at image coordinates "xyz".

        """
        if type(self.labels) == dict:
            return 0 if xyz not in self.labels.keys() else self.labels[xyz]
        elif type(self.labels) == ts.TensorStore:
            return int(self.labels[xyz].read().result())
        else:
            return self.labels[xyz]

    def count_edges(self):
        """
        Counts number of edges in all graphs in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        int

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

    def log(self, graph, edge_list):
        """
        Logs xyz coordinates of mistake and swc ids in a list if "txt_log"
        or writes an swc file if "log_swc".

        Parameters
        ----------
        graph : networkx.Graph
            Graph that contains edges in "edge_list".
        edge_list : list[tuple]
            Edges that correspond to a mistake.

        Returns
        -------
        None

        """
        if self.swc_log:
            self.swc_logger(graph, edge_list)

        if self.txt_log:
            self.txt_logger(graph, edge_list)

    def swc_logger(self, graph, edge_list):
        """
        Logs mistakes in an swc file.

        Parameters
        ----------
        graph : networkx.graph
            Graph that contains edges in "edge_list".
        edge_list : list[tuple]
            Edges that correspond to a mistake.

        Returns
        -------
        None

        """
        red = " 1.0 0.0 0.0"
        entries = swc_utils.make_entries(graph, edge_list, self.anisotropy)
        fn = self.prefix + str(self.site_cnt) + ".swc"
        path = os.path.join(self.swc_dir, fn)
        swc_utils.write_swc(path, entries, color=red)

    def txt_logger(self, graph, edge_list):
        """
        Logs xyz coordinates of mistake and swc ids in a list.

        Parameters
        ----------
        graph : networkx.graph
            Graph that contains edges in "edge_list".
        edge_list : list[tuple]
            Edges that correspond to a mistake.
        """
        for pair in edge_list:
            xyz_str = ""
            labels_str = ""
            for k in pair:
                xyz = swc_utils.node_to_world(graph, k, self.anisotropy)
                xyz_str += str(xyz) + ", "
                labels_str += str(self.get_label(graph, k)) + ", "
            self.mistakes_log.append(xyz_str + labels_str)

    def write_results(self, fn):
        """
        Writes "self.txt_log" to local machine at "self.log_dir".

        Parameters
        ----------
        fn : str
            Filename of text log.
        edge_list : list[tuple]
            List of edges that correspond to a mistakes.

        Returns
        -------
        None.

        """
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
