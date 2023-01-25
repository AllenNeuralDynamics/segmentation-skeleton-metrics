# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import networkx as nx

import aind_segmentation_evaluation.seg_metrics as sm
import aind_segmentation_evaluation.utils as utils
from aind_segmentation_evaluation.graph_routines import volume_to_dict


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

        # Initialize output_dir (if applicable)
        if output in ["swc"]:
            output_dir = os.path.join(output_dir, "merges")
            utils.mkdir(output_dir)

        # Initialize mistake counters
        self.site_cnt = 0
        self.edge_cnt = 0
        self.edge_list = {}

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
            dfs_edges = list(nx.dfs_edges(graph, 1))
            while len(dfs_edges) > 0:
                # Extract edge info
                (i, j) = dfs_edges.pop(0)
                val_i, val_j = utils.get_edge_values(
                    self.volume, graph, (i, j)
                )

                # Check for mistake
                if super().check_simple_mistake(val_i, val_j):
                    self.site_cnt += 1
                    fn = "merge_site-" + str(self.site_cnt) + ".swc"
                    super().log_simple_mistake(graph, i, fn)
                    dfs_edges, cnt1 = self.explore_merge(
                        graph, dfs_edges, i, val_i
                    )
                    dfs_edges, cnt2 = self.explore_merge(
                        graph, dfs_edges, j, val_j
                    )
                    self.edge_cnt += min(cnt1, cnt2)
                elif super().check_complex_mistake(val_i, val_j):
                    dfs_edges = self.process_complex_mistake(
                        graph, dfs_edges, (i, j)
                    )

            # Save Results
            super().write_results("merge_")

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
        merged_edges = list()
        visited = set()
        queue = [(-1, root)]  # parent, child
        while len(queue) > 0:
            # Visit
            i, j = queue.pop(0)
            val_i = utils.get_value(self.volume, graph, j)
            if val_i == val and i != -1:
                dfs_edges = utils.remove_edge(dfs_edges, (i, j))
                self.edge_list.update({(i, j), (j, i)})
                merged_edges.append((i, j))
            visited.add(j)

            # Populate queue
            for k in utils.get_nbs(graph, j):
                condition1 = k not in visited
                condition2 = (j, k) not in self.edge_list
                if condition1 and condition2:
                    queue.append((j, k))

        # Finalizations
        if len(merged_edges) > 0:
            fn = "merged_edges-" + str(self.site_cnt) + ".swc"
            super().log_complex_mistake(graph, merged_edges, root, fn)

        return dfs_edges, len(merged_edges)

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
        queue = [root_edge]
        visited = [root]
        while len(queue) > 0:
            # Visit
            (i, j) = queue.pop(0)
            val_j = utils.get_value(self.volume, graph, j)
            if super().check_simple_mistake(val, val_j):
                print("complex mistake")
                dfs_edges, cnt1 = self.explore_merge(
                    graph, dfs_edges, root, val
                )
                dfs_edges, cnt2 = self.explore_merge(
                    graph, dfs_edges, j, val_j
                )
                self.edge_cnt += cnt1 + cnt2  # min(cnt1, cnt2)
                self.site_cnt += 1

                fn = "merge_site-" + str(self.site_cnt) + ".swc"
                super().log_simple_mistake(graph, root, fn)

            # Add nbs to queue
            for k in utils.get_nbs(graph, j):
                val_k = utils.get_value(self.volume, graph, k)
                if k not in visited and val_k == 0:
                    queue.append((j, k))

            # Finish visit
            dfs_edges = utils.remove_edge(dfs_edges, (i, j))
            visited.append(j)

        return dfs_edges
