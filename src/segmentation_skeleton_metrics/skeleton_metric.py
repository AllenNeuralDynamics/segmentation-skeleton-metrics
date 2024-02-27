# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import KDTree

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import utils
from segmentation_skeleton_metrics.swc_utils import save, to_graph


class SkeletonMetric:
    """
    Class that evaluates the quality of a predicted segmentation by comparing
    the ground truth skeletons to the predicted segmentation mask. The
    accuracy is then quantified by detecting splits and merges, then computing
    the following metrics:
        (1) Number of splits
        (2) Number of merges
        (3) Percentage of omit edges
        (4) Percentage of merged edges
        (5) Edge accuracy
        (6) Expected Run Length (ERL)

    """

    def __init__(
        self,
        swc_paths,
        labels,
        anisotropy=[1.0, 1.0, 1.0],
        ignore_boundary_mistakes=False,
        black_holes_xyz_id=None,
        black_hole_radius=24,
        equivalent_ids=None,
        valid_ids=None,
        write_to_swc=False,
        output_dir=None,
    ):
        """
        Constructs skeleton metric object that evaluates the quality of a
        predicted segmentation.

        Parameters
        ----------
        swc_paths : list[str]
            List of paths to swc files such that each file corresponds to a
            neuron in the ground truth.
        labels : numpy.ndarray or tensorstore.TensorStore
            Predicted segmentation mask.
        anisotropy : list[float], optional
            Image to real-world coordinates scaling factors applied to swc
            files. The default is [1.0, 1.0, 1.0]
        black_holes_xyz_id : list
            ...
        black_hole_radius : float
            ...
        equivalent_ids : ...
            ...
        valid_ids : set
            ...

        Returns
        -------
        None.

        """
        # Store label options
        self.valid_ids = valid_ids
        self.labels = labels

        self.anisotropy = anisotropy
        self.ignore_boundary_mistakes = ignore_boundary_mistakes
        self.init_black_holes(black_holes_xyz_id)
        self.black_hole_radius = black_hole_radius

        self.write_to_swc = write_to_swc
        self.output_dir = output_dir

        # Build Graphs
        self.init_target_graphs(swc_paths, anisotropy)
        self.init_pred_graphs()

    def init_black_holes(self, black_holes):
        if black_holes:
            black_holes_xyz = [bh_dict["xyz"] for bh_dict in black_holes]
            black_holes_id = [bh_dict["swc_id"] for bh_dict in black_holes]
            self.black_holes = KDTree(black_holes_xyz)
            self.black_hole_labels = set(black_holes_id)
        else:
            self.black_holes = None
            self.black_hole_labels = set()

    def in_black_hole(self, xyz, print_nn=False):
        # Check whether black_holes exists
        if self.black_holes is None:
            return False

        # Search black_holes
        radius = self.black_hole_radius
        pts = self.black_holes.query_ball_point(xyz, radius)
        if print_nn:
            dd, ii = self.black_holes.query([xyz], k=[1])
            print("Nearest neighbor:", dd)
        if len(pts) > 0:
            return True
        else:
            return False

    def init_target_graphs(self, paths, anisotropy):
        """
        Initializes "self.target_graphs" by iterating over "paths" which
        correspond to neurons in the ground truth.

        Parameters
        ----------
        paths : list[str]
            List of paths to swc files which correspond to neurons in the
            ground truth.
        anisotropy : list[float]
            Image to real-world coordinates scaling factors applied to swc
            files.

        Returns
        -------
        None

        """
        self.target_graphs = dict()
        for path in paths:
            swc_id = os.path.basename(path).replace(".swc", "")
            self.target_graphs[swc_id] = to_graph(path, anisotropy=anisotropy)

    def init_pred_graphs(self):
        """
        Initializes "self.pred_graphs" by copying each graph in
        "self.target_graphs", then labels each node with the label in
        "self.labels" that coincides with it.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("Labelling Target Graphs...")
        t0 = time()
        self.pred_graphs = dict()
        self.label_to_node = dict()
        for cnt, (swc_id, graph) in enumerate(self.target_graphs.items()):
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            pred_graph, label_to_node = self.label_graph(graph)
            self.pred_graphs[swc_id] = pred_graph
            self.label_to_node[swc_id] = label_to_node

        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def label_graph(self, target_graph):
        """
        Iterates over nodes in "target_graph" and stores the label in the
        predicted segmentation mask (i.e. "self.labels") which coincides with
        each node as a node-level attribute called "pred_id".

        Parameters
        ----------
        target_graph : networkx.Graph
            Graph that represents a neuron from the ground truth.

        Returns
        -------
        target_graph : networkx.Graph
            Updated graph with node-level attributes called "pred_id".

        """
        pred_graph = nx.Graph(target_graph)
        label_to_node = dict()
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = []
            for i in pred_graph.nodes:
                img_coord = gutils.get_coord(pred_graph, i)
                threads.append(executor.submit(self.get_label, img_coord, i))

            # Store results
            for thread in as_completed(threads):
                i, label = thread.result()
                pred_graph.nodes[i].update({"pred_id": label})
                if label in label_to_node.keys():
                    label_to_node[label].add(i)
                else:
                    label_to_node[label] = set([i])
        return pred_graph, label_to_node

    def get_label(self, img_coord, return_node=False):
        """
        Gets label of voxel at "img_coord".

        Parameters
        ----------
        img_coord : numpy.ndarray
            Image coordinate of voxel to be read.

        Returns
        -------
        int
           Label of voxel at "img_coord".

        """
        label = self.__read_label(img_coord)
        if self.in_black_hole(img_coord):
            label = -1
        return self.output_label(label, return_node)

    def output_label(self, label, return_node):
        if return_node:
            return return_node, self.validate_label(label)
        else:
            return self.validate_label(label)

    def __read_label(self, coord):
        """
        Gets label at image coordinates "xyz".

        Parameters
        ----------
        coord : tuple[int]
            Coordinates that index into "self.labels".

        Returns
        -------
        int
           Label at image coordinates "xyz".

        """
        if type(self.labels) == ts.TensorStore:
            return int(self.labels[coord].read().result())
        else:
            return self.labels[coord]

    def validate_label(self, label):
        """
        Validates label by checking whether it is contained in
        "self.valid_ids".

        Parameters
        ----------
        label : int
            Label to be validated.

        Returns
        -------
        label : int
            There are two possibilities: (1) original label if either "label"
            is contained in "self.valid_ids" or "self.valid_labels" is None,
            or (2) 0 if "label" is not contained in self.valid_ids.

        """
        if self.valid_ids:
            if label not in self.valid_ids:
                return 0
        return label

    def compute_metrics(self):
        """
        Computes skeleton-based metrics.

        Parameters
        ----------
        None

        Returns
        -------
        ...

        """
        # Split evaluation
        print("Detecting Splits...")
        self.detect_splits()
        self.quantify_splits()

        # Merge evaluation
        print("Detecting Merges...")
        self.detect_merges()
        self.quantify_merges()

        # Compute metrics
        full_results, avg_results = self.compile_results()
        return full_results, avg_results

    def get_pred_ids(self, swc_id):
        """
        Gets the predicted label ids that intersect with the target graph
        corresponding to "swc_id".

        Parameters
        ----------
        swc_id : str

        """
        return set(self.label_to_node[swc_id].keys())

    def detect_splits(self):
        """
        Detects splits in the predicted segmentation, then deletes node and
        edges in "self.pred_graphs" that correspond to a split.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        t0 = time()
        target_graphs = self.target_graphs.items()
        for cnt, (swc_id, target_graph) in enumerate(target_graphs):
            # Initializations
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            pred_graph = self.pred_graphs[swc_id]

            # Run dfs
            r = gutils.sample_leaf(target_graph)
            dfs_edges = list(nx.dfs_edges(target_graph, source=r))
            while len(dfs_edges) > 0:
                # Visit edge
                (i, j) = dfs_edges.pop(0)
                label_i = pred_graph.nodes[i]["pred_id"]
                label_j = pred_graph.nodes[j]["pred_id"]
                if is_split(label_i, label_j):
                    dfs_edges, pred_graph = self.is_nonzero_misalignment(
                        target_graph, pred_graph, dfs_edges, i, j
                    )
                elif label_j == 0 or label_j == -1:
                    dfs_edges, pred_graph = self.is_zero_misalignment(
                        target_graph, pred_graph, dfs_edges, i, j
                    )

            # Update predicted graph
            pred_graph = gutils.delete_nodes(pred_graph, 0)
            pred_graph = gutils.delete_nodes(pred_graph, -1)
            label_to_node = gutils.store_labels(pred_graph)
            self.pred_graphs[swc_id] = pred_graph
            self.label_to_node[swc_id] = label_to_node

        # Report runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def is_zero_misalignment(
        self, target_graph, pred_graph, dfs_edges, nb, root
    ):
        """
        Determines whether zero-valued labels correspond to a split or
        misalignment between "target_graph" and the predicted segmentation
        mask.

        Parameters
        ----------
        target_graph : networkx.Graph
            ...
        pred_graph : networkx.Graph
            ...
        dfs_edges : list[tuple]
            List of edges to be processed for split detection.
        nb : int
            Neighbor of "root".
        root : int
            Node where possible split starts (i.e. zero-valued label).

        Returns
        -------
        dfs_edges : list[tuple].
            Updated "dfs_edges" with visited edges removed.
        pred_graph : networkx.Graph
            ...

        """
        # Search
        black_hole = False
        collision_labels = set()
        queue = [root]
        visited = set()
        while len(queue) > 0:
            j = queue.pop(0)
            label_j = pred_graph.nodes[j]["pred_id"]
            visited.add(j)
            if label_j > 0:
                collision_labels.add(label_j)
            else:
                # Check for black hole
                if label_j == -1:
                    black_hole = True

                # Add nbs to queue
                nbs = target_graph.neighbors(j)
                for k in [k for k in nbs if k not in visited]:
                    if utils.check_edge(dfs_edges, (j, k)):
                        queue.append(k)
                        dfs_edges = remove_edge(dfs_edges, (j, k))

        # Upd zero nodes
        if len(collision_labels) == 1 and not black_hole:
            label = collision_labels.pop()
            pred_graph = gutils.upd_labels(pred_graph, visited, label)

        return dfs_edges, pred_graph

    def is_nonzero_misalignment(
        self, target_graph, pred_graph, dfs_edges, nb, root
    ):
        # Initialize
        origin_label = pred_graph.nodes[nb]["pred_id"]
        hit_label = pred_graph.nodes[root]["pred_id"]
        parent = nb
        depth = 0

        # Search
        queue = [root]
        visited = set([nb])
        while len(queue) > 0:
            j = queue.pop(0)
            label_j = pred_graph.nodes[j]["pred_id"]
            visited.add(j)
            depth += 1
            if label_j == origin_label:
                # misalignment
                pred_graph = gutils.upd_labels(
                    pred_graph, visited, origin_label
                )
                return dfs_edges, pred_graph
            elif label_j == hit_label and depth < 16:
                # continue search
                nbs = list(target_graph.neighbors(j))
                nbs.remove(parent)
                if len(nbs) == 1:
                    if utils.check_edge(dfs_edges, (j, nbs[0])):
                        parent = j
                        queue.append(nbs[0])
                        dfs_edges = remove_edge(dfs_edges, (j, nbs[0]))
                else:
                    pred_graph = gutils.remove_edge(pred_graph, nb, root)
                    return dfs_edges, pred_graph
            else:
                # left hit label
                dfs_edges.insert(0, (parent, j))
                pred_graph = gutils.remove_edge(pred_graph, nb, root)
                return dfs_edges, pred_graph

    def quantify_splits(self):
        """
        Counts the number of splits, number of omit edges, and percent of omit
        edges for each graph in "self.pred_graphs".

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.split_cnts = dict()
        self.omit_cnts = dict()
        self.omit_percents = dict()
        for swc_id in self.target_graphs.keys():
            n_splits = gutils.count_splits(self.pred_graphs[swc_id])
            n_pred_edges = self.pred_graphs[swc_id].number_of_edges()
            n_target_edges = self.target_graphs[swc_id].number_of_edges()

            self.split_cnts[swc_id] = n_splits
            self.omit_cnts[swc_id] = n_target_edges - n_pred_edges
            self.omit_percents[swc_id] = 1 - n_pred_edges / n_target_edges

    def detect_merges(self):
        """
        Detects merges in the predicted segmentation, then deletes node and
        edges in "self.pred_graphs" that correspond to a merge.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initilize counts
        self.merge_cnts = self.init_merge_counter()
        self.merged_cnts = self.init_merge_counter()
        self.merged_percents = self.init_merge_counter()

        # Run detection
        t0 = time()
        for cnt, swc_id_1 in enumerate(self.pred_graphs.keys()):
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            for swc_id_2 in self.pred_graphs.keys():
                # Check if identical
                if swc_id_1 == swc_id_2:
                    continue

                # Compare pred_ids contained in graph
                pred_ids_1 = self.get_pred_ids(swc_id_1)
                pred_ids_2 = self.get_pred_ids(swc_id_2)
                intersection = pred_ids_1.intersection(pred_ids_2)
                for label in intersection:
                    #merged_1 = self.label_to_node[swc_id_1][label]
                    #merged_2 = self.label_to_node[swc_id_2][label]
                    # too_small = min(len(merged_1), len(merged_2)) > 16
                    if True:  # not too_small:
                        sites, dist = self.localize(swc_id_1, swc_id_2, label)
                        xyz = utils.get_midpoint(sites[0], sites[1])
                        if dist > 20 and not self.near_bdd(xyz):
                            # Write site to swc
                            if self.write_to_swc:
                                self.save_swc(sites[0], sites[1], "merge")

                            # Process merge
                            self.process_merge(swc_id_1, label)
                            self.process_merge(swc_id_2, label)

                    # Remove label to avoid reprocessing
                    del self.label_to_node[swc_id_1][label]
                    del self.label_to_node[swc_id_2][label]

        # Report Runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def localize(self, swc_id_1, swc_id_2, label):
        # Get merged nodes
        merged_1 = self.label_to_node[swc_id_1][label]
        merged_2 = self.label_to_node[swc_id_2][label]

        # Find closest pair
        min_dist = np.inf
        xyz_pair = [None, None]
        for i in merged_1:
            for j in merged_2:
                xyz_1 = self.target_graphs[swc_id_1].nodes[i]["xyz"]
                xyz_2 = self.target_graphs[swc_id_2].nodes[j]["xyz"]
                dist = utils.dist(xyz_1, xyz_2)
                if dist < min_dist:
                    min_dist = dist
                    xyz_pair = [xyz_1, xyz_2]
        return xyz_pair, min_dist

    def near_bdd(self, xyz_pair):
        near_bdd_bool = False
        if self.ignore_boundary_mistakes:
            merge_site = utils.get_midpoint(xyz_pair[0], xyz_pair[1])
            dims = self.labels.shape
            above = [merge_site[i] > dims[i] - 32 for i in range(3)]
            below = [merge_site[i] < 32 for i in range(3)]
            near_bdd_bool = True if any(above) or any(below) else False
        return near_bdd_bool

    def init_merge_counter(self):
        """
        Initializes a dictionary that is used to count the number of merge
        type mistakes for each pred_graph.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary used to count number of merge type mistakes.

        """
        return dict([(swc_id, 0) for swc_id in self.pred_graphs.keys()])

    def process_merge(self, swc_id, label):
        """
        Once a merge has been detected that corresponds to "label", every node
        in "self.pred_graph[swc_id]" with that label is deleted.

        Parameters
        ----------
        swc_id : str
            Key associated with the pred_graph to be searched.
        label : int
            Label assocatied with a merge.

        Returns
        -------
        None

        """
        # Update graph
        graph = self.pred_graphs[swc_id].copy()
        graph, merged_cnt = gutils.delete_nodes(graph, label, return_cnt=True)
        self.pred_graphs[swc_id] = graph

        # Update cnts
        self.merge_cnts[swc_id] += 1
        self.merged_cnts[swc_id] += merged_cnt

    def quantify_merges(self):
        """
        Computes the percentage of merged edges for each pred_graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.merged_percents = dict()
        for swc_id in self.target_graphs.keys():
            n_edges = self.target_graphs[swc_id].number_of_edges()
            self.merged_percents[swc_id] = self.merged_cnts[swc_id] / n_edges

    def compile_results(self):
        """
        Compiles a dictionary containing the metrics computed by this module.

        Parameters
        ----------
        None

        Returns
        -------
        full_results : dict
            Dictionary where the keys are swc_ids and the values are the result
            of computing each metric for the corresponding graphs.
        avg_result : dict
            Dictionary where the keys are names of metrics computed by this
            module and values are the averaged result over all swc_ids.

        """
        # Compute remaining metrics
        self.compute_edge_accuracy()
        self.compute_erl()

        # Summarize results
        swc_ids, results = self.generate_report()
        avg_results = dict([(k, np.mean(v)) for k, v in results.items()])
        avg_results["# merges"] = avg_results["# merges"] / 2

        # Reformat full results
        full_results = dict()
        for i, swc_id in enumerate(swc_ids):
            full_results[swc_id] = dict(
                [(key, results[key][i]) for key in results.keys()]
            )

        return full_results, avg_results

    def generate_report(self):
        """
        Generates a report by creating a list of the results for each metric.
        Each item in this list corresponds to a graph in "self.pred_graphs"
        and this list is ordered with respect to "swc_ids".

        Parameters
        ----------
        None

        Results
        -------
        swc_ids : list[str]
            Specifies the ordering of results for each value in "stats".
        stats : dict
            Dictionary where the keys are metrics and values are the result of
            computing that metric for each graph in "self.pred_graphs".

        """
        swc_ids = list(self.pred_graphs.keys())
        swc_ids.sort()
        stats = {
            "# splits": generate_result(swc_ids, self.split_cnts),
            "# merges": generate_result(swc_ids, self.merge_cnts),
            "% omit edges": generate_result(swc_ids, self.omit_percents),
            "% merged edges": generate_result(swc_ids, self.merged_percents),
            "edge accuracy": generate_result(swc_ids, self.edge_accuracy),
            "erl": generate_result(swc_ids, self.erl),
            "normalized erl": generate_result(swc_ids, self.normalized_erl),
        }
        return swc_ids, stats

    def compute_edge_accuracy(self):
        """
        Computes the edge accuracy of each pred_graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.edge_accuracy = dict()
        for swc_id in self.target_graphs.keys():
            omit_percent = self.omit_percents[swc_id]
            merged_percent = self.merged_percents[swc_id]
            self.edge_accuracy[swc_id] = 1 - omit_percent - merged_percent

    def compute_erl(self):
        """
        Computes the expected run length (ERL) of each pred_graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.erl = dict()
        self.normalized_erl = dict()
        for swc_id in self.target_graphs.keys():
            pred_graph = self.pred_graphs[swc_id]
            target_graph = self.target_graphs[swc_id]

            path_length = gutils.compute_path_length(target_graph)
            path_lengths = gutils.compute_run_lengths(pred_graph)
            wgts = path_lengths / max(np.sum(path_lengths), 1)

            self.erl[swc_id] = np.sum(wgts * path_lengths)
            self.normalized_erl[swc_id] = self.erl[swc_id] / path_length

    def list_metrics(self):
        """
        Lists metrics that are computed by this module.

        Parameters
        ----------
        None

        Returns
        -------
        metrics : list[str]
            List of metrics computed by this module.

        """
        metrics = [
            "# splits",
            "# merges",
            "% omit edges",
            "% merged edges",
            "edge accuracy",
            "erl",
            "normalized erl",
        ]
        return metrics

    def save_swc(self, xyz_1, xyz_2, mistake_type):
        xyz_1 = utils.to_world(xyz_1, self.anisotropy)
        xyz_2 = utils.to_world(xyz_2, self.anisotropy)
        if mistake_type == "split":
            color = "1.0 0.0 0.0"
            cnt = 1 + np.sum(list(self.split_cnts.values())) // 2
        else:
            color = "0.0 1.0 0.0"
            cnt = 1 + np.sum(list(self.merge_cnts.values())) // 2

        path = f"{self.output_dir}/{mistake_type}-{cnt}.swc"
        save(path, xyz_1, xyz_2, color=color)


# -- utils --
def is_split(a, b):
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
        Indication of whether there is a split.

    """
    return (a > 0 and b > 0) and (a != b)


def remove_edge(dfs_edges, edge):
    """
    Checks whether "edge" is in "dfs_edges" and removes it.

    Parameters
    ----------
    dfs_edges : list or set
        List or set of edges.
    edge : tuple
        Edge.

    Returns
    -------
    edges : list or set
        Updated list or set of edges with "dfs_edges" removed if it was
        contained in "dfs_edges".

    """
    if edge in dfs_edges:
        dfs_edges.remove(edge)
    elif (edge[1], edge[0]) in dfs_edges:
        dfs_edges.remove((edge[1], edge[0]))
    return dfs_edges


def generate_result(swc_ids, stats):
    """
    Reorders items in "stats" with respect to the order defined by "swc_ids".

    Parameters
    ----------
    swc_ids : list[str]
        List of all swc_ids of graphs in "self.pred_graphs".
    stats : dict
        Dictionary where the keys are swc_ids and values are the result of
        computing some metrics.

    Returns
    -------
    list
        Reorded items in "stats" with respect to the order defined by
        "swc_ids".

    """
    return [stats[swc_id] for swc_id in swc_ids]
