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
from aind_segmentation_evaluation.graph_routines import volume_to_dict, prune


class MergeMetric(sm.SegmentationMetrics):
    """
    Class that evaluates the quality of a segmentation in terms of the
    number of merges.
    """

    def __init__(
        self,
        shape,
        target_volume=None,
        path_to_target_volume=None,
        target_graphs=None,
        target_graphs_dir=None,
        pred_graphs=None,
        pred_graphs_dir=None,
        pred_volume=None,
        path_to_pred_volume=None,
        output=None,
        output_dir=None,
    ):
        """
        Constructs an object that evaluates a predicted segmentation mask in
        terms of the number of merges. Here are some additional details about
        the inputs:

        (1) At least one of {target_graphs, target_graphs_dir, target_volume,
        path_to_target_volume} must be provided, the recommended input is
        either "target_graphs_dir" or "target_graphs".

        (2) At least one of {pred_volume, path_to_pred_volume, pred_graphs,
        pred_graphs_dir} must be provided. The recommended input is either
        "pred_graphs" or "pred_graphs_dir".

        Parameters
        ----------
        shape : tuple
            Dimensions of image volume.
        target_volume : np.array, optional
            Target segmentation mask.
            The default is None.
        path_to_target_volume : str, optional
            Path to target segmentation mask (i.e. tif file).
            The default is None.
        target_graphs : list[networkx.Graph], optional
            List of graphs corresponding to target segmentation.
            The default is None.
        target_graph_dir : str, optional
            Path to directory containing swc files of target segmentation.
            The default is None.
        pred_graphs : list[nx.Graph], optional
            List of graphs corresponding to the predicted segmentation mask.
            The default is None.
        pred_graph_dir : str, optional
            Path to directory with swc files of predicted segmentation mask.
            The default is None.
        pred_volume : np.array, optional
            Predicted segmentation mask.
            The default is None.
        path_to_pred_volume : str, optional
            Path to predicted segmentation mask (i.e. tif file).
            The default is None.
        output : str, optional
            Type of output, supported options include 'swc' and 'tif'.
            The default is None.
        output_dir : str, optional
            Path to directory that outputs are written to.
            The default is None.

        Returns
        -------
        None.

        """
        # Upload data
        self.shape = shape
        if target_volume is None:
            target_volume = super().init_volume(
                path_to_target_volume, target_graphs, target_graphs_dir
            )
        else:
            target_volume = volume_to_dict(target_volume)

        if pred_graphs is None:
            pred_graphs = super().init_graphs(
                pred_graphs_dir, pred_volume, path_to_pred_volume
            )
            upd_pred_graphs = []
            for graph in pred_graphs:
                graph = prune(graph)
                if graph.number_of_nodes() > 1:
                    upd_pred_graphs.append(graph)
            pred_graphs = upd_pred_graphs

        # Initialize output_dir (if applicable)
        if output in ["swc"]:
            output_dir = os.path.join(output_dir, "merges")
            utils.mkdir(output_dir)

        # Initialize mistake counters
        self.site_cnt = 0
        self.edge_cnt = 0
        self.interior_site_cnt = 0
        self.merged_edges = set()

        # Initialize for ERL
        self.run_lengths = dict()
        target_labels = np.unique(list(target_volume.values()))
        for i in [i for i in target_labels]:
            self.run_lengths[i] = []

        super().__init__(pred_graphs, target_volume, shape, output, output_dir)

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
                    self.volume, graph, (i, j)
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
            val_i = utils.get_value(self.volume, graph, j)
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
        if utils.get_value(self.volume, graph, edge[0]) > 0:
            root_edge = edge
        else:
            root_edge = (edge[1], edge[0])
        root = root_edge[0]
        val = utils.get_value(self.volume, graph, root)

        # Main routine
        flag = False
        queue = [root_edge]
        visited = [root]
        while len(queue) > 0:
            # Visit
            (i, j) = queue.pop(0)
            val_j = utils.get_value(self.volume, graph, j)
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
