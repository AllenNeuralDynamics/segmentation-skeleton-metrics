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


class SplitMetric(sm.SegmentationMetrics):
    """
    Class that evaluates a predicted segmentation mask in terms of the
    number of splits.

    """

    def __init__(
        self,
        shape,
        target_graphs=None,
        target_graphs_dir=None,
        target_volume=None,
        path_to_target_volume=None,
        pred_volume=None,
        path_to_pred_volume=None,
        pred_graphs=None,
        pred_graphs_dir=None,
        output=None,
        output_dir=None,
    ):
        """
        Constructs an object that evaluates a predicted segmentation mask in
        terms of the number of splits. Here are some additional details about
        the inputs:

        (1) At least one of {target_graphs, target_graphs_dir, target_volume,
        path_to_target_volume} must be provided, the recommended input is
        either "target_graphs" or "target_graphs_dir".

        (2) At least one of {pred_volume, path_to_pred_volume, pred_graphs,
        pred_graphs_dir} must be provided. The recommended input is either
        "pred_volume" or "path_to_pred_volume".

        Parameters
        ----------
        shape : tuple
            Dimensions of image volume.
        target_graphs : list[networkx.Graph], optional
            List of graphs corresponding to target segmentation.
            The default is None.
        target_graph_dir : str, optional
            Path to directory containing the swc files of target segmentation.
            The default is None.
        target_volume : np.array, optional
            Target segmentation mask.
            The default is None.
        path_to_target_volume : str, optional
            Path to target segmentation mask (i.e. tif file).
            The default is None.
        pred_volume : np.array, optional
            Predicted segmentation mask.
            The default is None.
        path_to_pred_volume : str, optional
            Path to predicted segmentation mask (i.e. tif file).
            The default is None.
        pred_graphs : list[nx.Graph], optional
            List of graphs corresponding to the predicted segmentation mask.
            The default is None.
        pred_graph_dir : str, optional
            Path to directory with swc files of predicted segmentation mask.
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
        if target_graphs is None:
            target_graphs = super().init_graphs(
                target_graphs_dir, target_volume, path_to_target_volume
            )

        if pred_volume is None:
            pred_volume = super().init_volume(
                path_to_pred_volume, pred_graphs, pred_graphs_dir
            )
        else:
            pred_volume = volume_to_dict(pred_volume)

        # Initialize output_dir (if applicable)
        if output in ["swc"]:
            output_dir = os.path.join(output_dir, "splits")
            utils.mkdir(output_dir)

        # Initialize counters
        self.edge_cnt = 0
        self.site_cnt = 0

        super().__init__(target_graphs, pred_volume, shape, output, output_dir)

    def detect_mistakes(self):
        """
        Detects splits in the predicted segmentation mask.

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
            dfs_edges = list(nx.dfs_edges(graph, source=1))
            while len(dfs_edges) > 0:
                # Extract edge info
                (i, j) = dfs_edges.pop(0)
                val_i, val_j = utils.get_edge_values(
                    self.volume, graph, (i, j)
                )

                # Check missing neuron flag
                if val_i > 0:
                    miss_flag = False

                # Check for mistake
                if super().check_simple_mistake(val_i, val_j):
                    self.site_cnt += 1
                    self.edge_cnt += 1
                    fn = "split_site-" + str(self.site_cnt) + ".swc"
                    super().log_simple_mistake(graph, i, fn)
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
            Filename of swc that will be written.

        Returns
        -------
        list[tuple].
            Updated "dfs_edges".

        """
        # Initializations
        if utils.get_value(self.volume, graph, root_edge[1]) > 0:
            root_edge = (root_edge[1], root_edge[0])

        root = root_edge[0]
        root_val = utils.get_value(self.volume, graph, root)

        # Main
        log_flag = False
        queue = [root_edge]
        visited_nodes = {root}
        visited_edges = []
        while len(queue) > 0:
            # Visit
            (i, j) = queue.pop(0)
            val = utils.get_value(self.volume, graph, j)
            add_nbs = False
            if super().check_simple_mistake(root_val, val):
                self.site_cnt += 1
                fn_a = "split_site-" + str(self.site_cnt + 1) + "a.swc"
                fn_b = "split_site-" + str(self.site_cnt + 1) + "b.swc"
                super().log_simple_mistake(graph, root, fn_a)
                super().log_simple_mistake(graph, j, fn_b)
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
