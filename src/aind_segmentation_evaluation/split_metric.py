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
from aind_segmentation_evaluation.graph_routines import swc_to_graph

from random import sample


class SplitMetric(sm.SegmentationMetrics):
    """
    Class that evaluates a predicted segmentation in terms of the number
    of splits.
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
        of the number of splits.

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
        if type(pred_labels) is str:
            pred_labels = super().init_labels(pred_labels, filetype)

        if type(target_graphs) is str:
            target_graphs = swc_to_graph(target_graphs, pred_labels.shape)

        if output in ["swc"]:
            output_dir = os.path.join(output_dir, "splits")

        super().__init__(target_graphs, pred_labels, output, output_dir)

    def detect_mistakes(self):
        """
        Detects splits in the predicted segmentation.

        Parameters
        -------
        None

        Returns
        -------
        None

        """
        # Detection
        for graph in self.graphs:
            miss_flag = True
            root = sample(list(graph.nodes), 1)[0]
            dfs_edges = list(nx.dfs_edges(graph, source=root))
            while len(dfs_edges) > 0:
                # Extract edge info
                (i, j) = dfs_edges.pop(0)
                val_i, val_j = utils.get_edge_values(
                    self.labels, graph, (i, j)
                )

                # Check missing neuron flag
                if val_i > 0:
                    miss_flag = False

                # Check for mistake
                if super().check_simple_mistake(val_i, val_j):
                    self.site_cnt += 1
                    self.edge_cnt += 1
                    fn = "split_site-" + str(self.site_cnt) + ".swc"
                    super().log_simple_mistake(graph, (i, j), fn)
                elif super().check_complex_mistake(val_i, val_j):
                    dfs_edges = self.process_complex_mistake(
                        graph, dfs_edges, (i, j)
                    )

            # Check whether whole neuron is missing
            if miss_flag:
                fn = "split_edges-" + str(self.site_cnt) + ".swc"
                list_of_edges = list(nx.dfs_edges(graph, source=1))
                super().log_complex_mistake(graph, list_of_edges, j, fn)
                self.edge_cnt += len(list_of_edges)

        # Save results
        super().write_results("split_")

    def process_complex_mistake(self, graph, dfs_edges, root_edge):
        """
        Determines whether complex mistake is a split.

        Parameters
        ----------
        dfs_edges : list[tuple]
            List of edges in order wrt a depth first search.
        root_edge : tuple
            Edge where possible split starts.
        fn : str
            Filename of swc that is to be written.

        Returns
        -------
        list[tuple].
            Updated "dfs_edges".

        """
        # Initializations
        if utils.get_value(self.labels, graph, root_edge[1]) > 0:
            root_edge = (root_edge[1], root_edge[0])

        root = root_edge[0]
        root_val = utils.get_value(self.labels, graph, root)

        # Main
        log_flag = False
        queue = [root_edge]
        visited_nodes = {root}
        visited_edges = []
        while len(queue) > 0:
            # Visit
            (i, j) = queue.pop(0)
            val = utils.get_value(self.labels, graph, j)
            add_nbs = False
            if super().check_simple_mistake(root_val, val):
                self.site_cnt += 1
                fn_a = "split_site-" + str(self.site_cnt + 1) + "a.swc"
                fn_b = "split_site-" + str(self.site_cnt + 1) + "b.swc"
                super().log_simple_mistake(graph, (root, j), fn_a)
                super().log_simple_mistake(graph, (root, j), fn_b)
                log_flag = True
            elif val == 0:
                add_nbs = True
                if graph.degree[j] == 1:
                    log_flag = True

            # Finish visit
            visited_nodes.add(j)
            visited_edges.append((i, j))
            dfs_edges = utils.remove_edge(dfs_edges, (i, j))

            # Check nbs
            if add_nbs:
                for k in utils.get_nbs(graph, j):
                    if k not in visited_nodes:
                        queue.append((j, k))

        # Finalizations
        if log_flag:
            fn = "split_edges-" + str(self.site_cnt) + ".swc"
            super().log_complex_mistake(graph, visited_edges, root, fn)
            self.edge_cnt += len(visited_edges)
        return dfs_edges
