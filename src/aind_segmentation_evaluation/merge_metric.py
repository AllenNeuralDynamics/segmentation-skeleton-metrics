# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
import networkx as nx
import numpy as np
import aind_segmentation_evaluation.seg_metrics as sm
import aind_segmentation_evaluation.utils as utils
from aind_segmentation_evaluation.graph_routines import swc_to_graph


class MergeMetric(sm.SegmentationMetrics):
    """
    Class that evaluates the quality of a segmentation in terms of the
    number of merges.
    """

    def __init__(
        self,
        target_graphs,
        target_labels,
        pred_graphs,
        pred_labels,
        filetype=None,
        output=None,
        output_dir=None,
        scaling_factors=[1.0, 1.0, 1.0],
    ):
        """
        Constructs an object that evaluates a predicted segmentation in terms
        of the number of merges.

        Parameters
        ----------
        target_graphs : list[networkx.Graph] or str
            List of graphs corresponding to target segmentation or path to
            directory of swc files.
        target_labels : np.array, n5, or str
            Target segmentation or path to target segmentation.
        pred_graphs : list[networkx.Graph] or str
            List of graphs corresponding to the predicted segmentation or path
            to directory of swc files.
        pred_labels : np.array, n5, or str
            Predicted segmentation or path to predicted segmentation.
        filetype : str, optional
            File type of target_labels and pred_labels. Supported file types
            include tif and n5. The default is None.
        output : str, optional
            Type of output, supported options include "swc" and "tif".
            The default is None.
        output_dir : str, optional
            Path to directory that outputs are written to.
            The default is None.
        scaling_factors : list[float], optional
            Scaling factor from image to real-world coordinates.
            The default is None.

        Returns
        -------
        None.

        """
        # Upload data
        if type(target_labels) is str:
            target_labels = super().init_labels(target_labels, filetype)

        if type(pred_graphs) is str:
            pred_graphs = swc_to_graph(pred_graphs, target_labels.shape)

        if output in ["swc"]:
            output_dir = os.path.join(output_dir, "merges")

        # Initialize for ERL
        self.merged_edges = set()
        self.run_lengths = dict()
        list_of_labels = np.unique(target_labels)
        for i in [i for i in list_of_labels if i != 0]:
            self.run_lengths[i] = []

        super().__init__(pred_graphs, target_labels, output, output_dir)

    def detect_mistakes(self):
        """
        Detects merges in the predicted segmentation.

        Parameters
        -------
        None

        Returns
        -------
        None

        """
        for graph in self.graphs:
            dfs_edges = list(nx.dfs_edges(graph))
            merge_flag = False
            nonzero_flag = False
            run_length = 0
            target_label = -1
            while len(dfs_edges) > 0:
                # Extract edge info
                (i, j) = dfs_edges.pop(0)
                val_i, val_j = utils.get_edge_values(
                    self.labels, graph, (i, j)
                )

                # Check for mistake
                if super().check_simple_mistake(val_i, val_j):
                    merge_flag = True
                    self.site_cnt += 1
                    fn = "merge_site-" + str(self.site_cnt) + ".swc"
                    super().log_simple_mistake(graph, (i, j), fn)
                    dfs_edges = self.explore_merge(graph, dfs_edges, i, val_i)
                    dfs_edges = self.explore_merge(graph, dfs_edges, j, val_j)
                elif super().check_complex_mistake(val_i, val_j):
                    dfs_edges, flag = self.process_complex_mistake(
                        graph, dfs_edges, (i, j)
                    )
                    merge_flag = merge_flag or flag
                elif val_i == val_j and val_i > 0:
                    run_length += 1

                # Check nonzero flag
                if (val_i != 0 or val_j != 0) and not nonzero_flag:
                    nonzero_flag = True
                    target_label = val_i if val_i != 0 else val_j

            # Store data for ERL
            if merge_flag:
                self.run_lengths[target_label].append(0)
            elif nonzero_flag:
                self.run_lengths[target_label].append(run_length)

        # Save Results
        super().write_results("merge_")
        self.edge_cnt = len(self.merged_edges) // 2

    def explore_merge(self, graph, dfs_edges, root, val):
        """
        Traverses "graph" from "root" to determine how many and which
        edges are merged.

        Parameters
        ----------
        graph : networkx.Graph
            Graph that represents a neuron.
        dfs_edges : list[tuple]
            List of edges in graph ordered wrt a depth first search.
        root : int
            Root node of list of edges in "dfs_edges".
        val : int
            Value of root node.

        Returns
        -------
        list[tuple]
            List of merged edges.

        """
        cur_merge = list()
        visited = set()
        queue = [(-1, root)]  # parent, child
        while len(queue) > 0:
            # Visit
            i, j = queue.pop(0)
            val_i = utils.get_value(self.labels, graph, j)
            if val_i == val and i != -1:
                self.merged_edges.update({(i, j), (j, i)})
                cur_merge.append((i, j))
                dfs_edges = utils.remove_edge(dfs_edges, (i, j))
            visited.add(j)

            # Populate queue
            for k in utils.get_nbs(graph, j):
                condition1 = k not in visited
                condition2 = (j, k) not in self.merged_edges
                if condition1 and condition2:
                    queue.append((j, k))

        # Finalizations ~ edit this to output merged edges
        if len(cur_merge) > 0:
            edge_cnt = len(self.merged_edges) // 2
            fn = "merged_edges-" + str(edge_cnt) + ".swc"
            super().log_complex_mistake(graph, cur_merge, root, fn)

        return dfs_edges

    def process_complex_mistake(self, graph, dfs_edges, edge):
        """
        Determines whether complex mistake is a merge.

        Parameters
        ----------
        graph : networkx.Graph
            Graph that represents a neuron.
        dfs_edges : list[tuple]
            List of edges in graph ordered wrt dfs.
        edge : tuple
            Edge where possible merge starts.

        Returns
        -------
        dfs_edges : list[tuple]
            List of edges in graph ordered wrt dfs.

        """
        # Initialize root edge
        if utils.get_value(self.labels, graph, edge[0]) > 0:
            root_edge = edge
        else:
            root_edge = (edge[1], edge[0])
        root = root_edge[0]
        val = utils.get_value(self.labels, graph, root)

        # Main routine
        flag = False
        queue = [root_edge]
        visited = [root]
        while len(queue) > 0:
            # Visit
            (i, j) = queue.pop(0)
            val_j = utils.get_value(self.labels, graph, j)
            if super().check_simple_mistake(val, val_j):
                flag = True
                self.site_cnt += 1
                dfs_edges = self.explore_merge(graph, dfs_edges, root, val)
                dfs_edges = self.explore_merge(graph, dfs_edges, j, val_j)
                fn = "merge_site-" + str(self.site_cnt) + ".swc"
                super().log_simple_mistake(graph, (root, j), fn)
            elif val_j == 0:
                for k in utils.get_nbs(graph, j):
                    if k not in visited:
                        queue.append((j, k))

            # Finish visit
            dfs_edges = utils.remove_edge(dfs_edges, (i, j))
            visited.append(j)

        return dfs_edges, flag
