# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import networkx as nx
from eval_seg.graph_routines import *
from eval_seg.seg_metrics import *
from eval_seg.utils import *


class MergeMetric(SegmentationMetrics):
    def __init__(
        self,
        shape,
        target_volume=None,
        path_to_target_volume=None,
        target_graphs_dir=None,
        pred_graphs=None,
        pred_graphs_dir=None,
        path_to_pred_volume=None,
        output=None,
        output_dir=None,
    ):
        """
        Constructs object that evaluates predicted segmentation in terms of the
        number of merges.

        Parameters
        ----------
        shape : tuple
            Dimensions of image volume.
        target_volume : np.array(), optional
            Target segmentation
        path_to_target_volume : str, optional
            Path to target volume (i.e. tif file). The default is None.
        target_graph_dir : str, optional
            Path to directory containing target swc files. The default is None.
        pred_graphs : list[nx.Graph()], optional
            List of graphs such that each graph corresponds to a predicted neuron.
        pred_graph_dir : str, optional
            Path to directory containing predicted swc files. The default is None.
        path_to_pred_volume : str, optional
            Path to predicted volume (i.e. tif file). The default is None.
        output : str, optional
            Type of output, supported options are tif, swc. The default is None.
        output_dir : str, optional
            Path to directory where outputs are written. The default is None.

        Returns
        -------
        None.

        """
        # Upload data
        self.shape = shape
        if target_volume == None:
            target_volume = super().init_volume(
                path_to_target_volume, target_graphs_dir
            )

        if pred_graphs == None:
            pred_graphs = super().init_graphs(
                pred_graphs_dir, path_to_pred_volume
            )

        # Initialize output_dir (if applicable)
        if output in ["swc"]:
            output_dir = os.path.join(output_dir, "merges")
            mkdir(output_dir)

        # Initialize counters
        self.merge_cnt = 0
        self.merge_edge_cnt = 0

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
                val_i, val_j = get_edge_values(self.volume, graph, (i, j))

                # Check for mistake
                if super().check_simple_mistake(val_i, val_j):
                    self.merge_cnt += 1
                    fn = "merge_site-" + str(self.merge_cnt) + ".swc"
                    super().log_simple_mistake(graph, i, fn)
                    dsf_edges, cnt1 = self.explore_merge(
                        graph, dfs_edges, i, val_i
                    )
                    dsf_edges, cnt2 = self.explore_merge(
                        graph, dfs_edges, j, val_j
                    )
                    self.merge_edge_cnt += min(cnt1, cnt2)
                elif super().check_complex_mistake(val_i, val_j):
                    dfs_edges = self.process_complex_mistake(
                        graph, dfs_edges, (i, j)
                    )

            # Save Results
            super().write_results("merge_")

    def explore_merge(self, graph, dfs_edges, root, val):
        """
        Traverses "graph" from "root" to determine how many edges are merged.

        Parameters
        ----------
        graph : networkx.Graph()
            Graph that represents a neuron..
        dfs_edges : list[tuple]
            List of edges in graph ordered wrt dfs.
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
        queue = [root]
        while len(queue) > 0:
            i = queue.pop(0)
            for j in [j for j in get_nbs(graph, i) if not j in visited]:
                # Visit
                if val == get_value(self.volume, graph, j):
                    merged_edges.append((i, j))
                    queue.append(j)

                # Finish visit
                visited.add(j)
                dfs_edges = remove_edge(dfs_edges, (i, j))

        # Finalizations
        if len(merged_edges) > 0:
            fn = "merged_edges-" + str(self.merge_cnt) + ".swc"
            super().log_complex_mistake(graph, merged_edges, root, fn)

        return dfs_edges, len(merged_edges)

    def process_complex_mistake(self, graph, dfs_edges, edge):
        """
        Determines whether complex mistake is a merge.

        Parameters
        ----------
        graph : networkx.Graph()
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
        if get_value(self.volume, graph, edge[0]) > 0:
            root_edge = edge
        else:
            root_edge = (edge[1], edge[0])
        root = root_edge[0]
        val = get_value(self.volume, graph, root)

        # Main routine
        queue = [root_edge]
        visited = [root]
        while len(queue) > 0:
            # Visit
            (i, j) = queue.pop(0)
            val_j = get_value(self.volume, graph, j)
            if super().check_simple_mistake(val, val_j):
                dsf_edges, cnt1 = self.explore_merge(
                    graph, dfs_edges, root, val
                )
                dsf_edges, cnt2 = self.explore_merge(
                    graph, dfs_edges, j, val_j
                )
                self.merge_edge_cnt += min(cnt1, cnt2)
                self.merge_cnt += 1

                fn = "merge_site-" + str(self.merge_cnt) + ".swc"
                super().log_simple_mistake(graph, root, fn)

            # Add nbs to queue
            for k in get_nbs(graph, j):
                val_k = get_value(self.volume, graph, k)
                if not k in visited and val_k == 0:
                    queue.append((j, k))

            # Finish visit
            dfs_edges = remove_edge(dfs_edges, (i, j))
            visited.append(j)

        return dfs_edges

    def compute_erl(self):
        pass
