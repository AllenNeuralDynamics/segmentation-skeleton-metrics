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

import aind_segmentation_evaluation.graph_routines as gr
import aind_segmentation_evaluation.utils as utils


SUPPORTED_FILETYPES = ["tif", "tiff", "n5"]


class SegmentationMetrics(ABC):
    """
    Class that evaluates a segmentation in terms of the number of
    splits and merges.

    """

    def __init__(self, graphs, labels, output, output_dir):
        """
        Constructs object that evaluates a segmentation mask.

        Parameters
        ----------
        graph : list[networkx.Graph]
            List of graphs where each graph represents a neuron.
        labels : dict
            Segmentation mask.
        output : str
            Type of output. Supported options include "swc" and "tif".
        output_dir : str
            Directory where "output" is written to.

        Returns
        -------
        None.

        """
        self.graphs = graphs
        self.labels = labels

        self.output = output
        self.output_dir = output_dir

        # Initialize mistake trackers
        self.site_cnt = 0
        self.edge_cnt = 0
        self.site_mask = np.zeros(labels.shape, dtype=bool)
        if self.output in ["tif", "tiff"]:
            self.edge_mask = np.zeros(labels.shape, dtype=bool)
        if self.output in ["swc"]:
            utils.mkdir(output_dir)

    def init_labels(self, path_to_labels, filetype):
        """
        Initializes a volume by uploading file with extension "filetype".

        Parameters
        ----------
        path_to_volume : str
            Path to image volume.
        file_type : str
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
            return utils.upload_tensorstore(path_to_labels)
        elif filetype == "n5":
            return utils.upload_n5(path_to_labels)
        else:
            return utils.upload_tif(path_to_labels)

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
        for graph in self.graphs:
            self.edge_cnt += graph.number_of_edges()

    def check_simple_mistake(self, a, b):
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
        return (a > 0 and b > 0) and (a != b)

    def check_complex_mistake(self, a, b):
        """
        Checks if one of "a" and "b" is positive and the other is zero-valued.

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
        condition_1 = (a > 0) and (b == 0)
        condition_2 = (b > 0) and (a == 0)
        return condition_1 or condition_2

    def log_simple_mistake(self, graph, node_tuple, fn):
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
        i = utils.get_idx(graph, node_tuple[0])
        j = utils.get_idx(graph, node_tuple[1])
        self.site_mask[i] = 1
        self.site_mask[j] = 1
        if self.output == "swc":
            red = " 1.0 0.0 0.0"
            xyz = utils.get_xyz(graph, i)
            list_of_entries = [gr.get_swc_entry(xyz, 8, -1)]
            path_to_swc = os.path.join(self.output_dir, fn)
            gr.write_swc(path_to_swc, list_of_entries, color=red)

    def log_complex_mistake(self, graph, list_of_edges, root, fn):
        """
        Logs list of xyz coordinates of mistake in a
        numpy.array or writes an swc file.

        Parameters
        ----------
        graph : networkx.Graph
            Graph that represents a neuron.
        list_of_edges : list[tuple]
            List of edges that form a path.
        root_edge : int
            Root node corresponding to "list_of_edges".
        fn: str
            Filename of swc that will be written.

        Returns
        -------
        None.

        """
        if self.output == "swc":
            red = " 1.0 0.0 0.0"
            reindex = {root: 1}
            xyz = utils.get_xyz(graph, root)
            swc = [gr.get_swc_entry(xyz, 8, -1)]
            for i, j in list_of_edges:
                xyz = utils.get_xyz(graph, j)
                swc.append(gr.get_swc_entry(xyz, 8, reindex[i]))
                reindex[j] = len(reindex) + 1
            path = os.path.join(self.output_dir, fn)
            gr.write_swc(path, swc, color=red)
        elif self.output == "tif":
            for i, j in list_of_edges:
                idx = utils.get_idx(graph, j)
                self.edge_mask[idx] = 1

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
        if self.output in ["tif", "tiff"]:
            path_to_site_mask = os.path.join(self.output_dir, fn + "sites.tif")
            path_to_edge_mask = os.path.join(self.output_dir, fn + "edges.tif")
            imwrite(path_to_site_mask, self.site_mask)
            imwrite(path_to_edge_mask, self.edge_mask)

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
    def process_complex_mistake(self):
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
