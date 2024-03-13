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
from segmentation_skeleton_metrics import split_detection
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
        pred_labels,
        pred_swc_paths,
        target_swc_paths,
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
        target_swc_paths : list[str]
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
        self.labels = pred_labels

        self.anisotropy = anisotropy
        self.ignore_boundary_mistakes = ignore_boundary_mistakes
        self.init_black_holes(black_holes_xyz_id)
        self.black_hole_radius = black_hole_radius

        self.write_to_swc = write_to_swc
        self.output_dir = output_dir

        # Build Graphs
        self.pred_graphs = self.init_graphs(pred_swc_paths, anisotropy)
        self.target_graphs = self.init_graphs(target_swc_paths, anisotropy)
        self.labeled_target_graphs = self.init_labeled_target_graphs()

        # Build kdtree
        self.init_xyz_to_swc_node()
        self.init_kdtree()

    def init_xyz_to_swc_node(self):
        self.xyz_to_swc_node = dict()
        for swc_id, graph in self.target_graphs.items():
            for i in graph.nodes:
                xyz = tuple(graph.nodes[i]["xyz"])
                if xyz in self.xyz_to_swc_node.keys():
                    self.xyz_to_swc_node[xyz][swc_id] = i
                else:
                    self.xyz_to_swc_node[xyz] = {swc_id: i}

    def init_kdtree(self):
        xyz_list = []
        for swc_id, graph in self.target_graphs.items():
            for i in graph.nodes:
                xyz_list.append(graph.nodes[i]["xyz"])
        self.target_graphs_kdtree = KDTree(xyz_list)

    def get_projection(self, xyz):
        d, idx = self.target_graphs_kdtree.query(xyz, k=1)
        return tuple(self.target_graphs_kdtree.data[idx]), d

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
        else:
            radius = self.black_hole_radius
            pts = self.black_holes.query_ball_point(xyz, radius)
            if len(pts) > 0:
                return True
            else:
                return False

    def init_graphs(self, paths, anisotropy):
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
        graphs = dict()
        for path in paths:
            swc_id = os.path.basename(path).replace(".swc", "")
            graphs[swc_id] = to_graph(path, anisotropy=anisotropy)
        return graphs

    def init_labeled_target_graphs(self):
        """
        Initializes "self.labeled_target_graphs" by copying each graph in
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
        labeled_target_graphs = dict()
        self.label_to_node = dict()
        for cnt, (swc_id, graph) in enumerate(self.target_graphs.items()):
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            pred_graph, label_to_node = self.label_graph(graph)
            labeled_target_graphs[swc_id] = pred_graph
            self.label_to_node[swc_id] = label_to_node

        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")
        return labeled_target_graphs

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
        edges in "self.labeled_target_graphs" that correspond to a split.

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
            # Detection
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            labeled_graph = self.labeled_target_graphs[swc_id]
            labeled_graph = split_detection.run(target_graph, labeled_graph)

            # Update predicted graph
            labeled_graph = gutils.delete_nodes(labeled_graph, 0)
            labeled_graph = gutils.delete_nodes(labeled_graph, -1)
            label_to_node = gutils.store_labels(labeled_graph)
            self.labeled_target_graphs[swc_id] = labeled_graph
            self.label_to_node[swc_id] = label_to_node

        # Report runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def quantify_splits(self):
        """
        Counts the number of splits, number of omit edges, and percent of omit
        edges for each graph in "self.labeled_target_graphs".

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
            n_splits = gutils.count_splits(self.labeled_target_graphs[swc_id])
            n_pred_edges = self.labeled_target_graphs[swc_id].number_of_edges()
            n_target_edges = self.target_graphs[swc_id].number_of_edges()

            self.split_cnts[swc_id] = n_splits
            self.omit_cnts[swc_id] = n_target_edges - n_pred_edges
            self.omit_percents[swc_id] = 1 - n_pred_edges / n_target_edges

    def detect_merges(self):
        """
        Detects merges in the predicted segmentation, then deletes node and
        edges in "self.labeled_target_graphs" that correspond to a merge.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initilize counts
        self.merge_cnts = self.init_counter()
        self.merged_cnts = self.init_counter()
        self.merged_percents = self.init_counter()

        # Run detection
        t0 = time()
        self.set_target_to_pred()
        for cnt, target_id_1 in enumerate(self.labeled_target_graphs.keys()):
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            for target_id_2 in self.labeled_target_graphs.keys():
                # Check if identical
                if target_id_1 == target_id_2:
                    continue

                # Compare pred_ids contained in graph
                pred_ids_1 = self.get_pred_ids(target_id_1)
                pred_ids_2 = self.get_pred_ids(target_id_2)
                intersection = pred_ids_1.intersection(pred_ids_2)
                for label in intersection:
                    valid_1 = label in self.target_to_pred[target_id_1]
                    valid_2 = label in self.target_to_pred[target_id_2]
                    if valid_1 and valid_2:
                        sites, d = self.localize(
                            target_id_1, target_id_2, label
                        )
                        if d < 30 and self.write_to_swc:
                            # Process merge
                            self.save_swc(sites[0], sites[1], "merge")
                            self.process_merge(target_id_1, label)
                            self.process_merge(target_id_2, label)

                            # Remove label to avoid reprocessing
                            del self.label_to_node[target_id_1][label]
                            del self.label_to_node[target_id_2][label]

        # Report Runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def set_target_to_pred(self):
        self.target_to_pred = self.init_tracker()
        for pred_id, graph in self.pred_graphs.items():
            # Compute intersections
            hit_target_ids = dict()
            hit_multilabels_xyz = set()
            for i in graph.nodes:
                xyz = tuple(graph.nodes[i]["xyz"])
                hat_xyz, d = self.get_projection(xyz)
                if d < 3:
                    target_ids = list(self.xyz_to_swc_node[hat_xyz].keys())
                    if len(target_ids) > 1:
                        hit_multilabels_xyz.add(hat_xyz)
                    else:
                        target_id = target_ids[0]
                        hat_i = self.xyz_to_swc_node[hat_xyz][target_id]
                        hit_target_ids = utils.append_dict_value(
                            hit_target_ids, target_id, hat_i
                        )

            # Process
            hit_target_ids = utils.resolve_multilabels(
                hit_multilabels_xyz, hit_target_ids, self.xyz_to_swc_node
            )
            for target_id, values in hit_target_ids.items():
                if len(values) > 16:
                    self.target_to_pred[target_id].add(int(pred_id))

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

    def near_bdd(self, xyz):
        near_bdd_bool = False
        if self.ignore_boundary_mistakes:
            above = [xyz[i] >= self.labels.shape[i] - 32 for i in range(3)]
            below = [xyz[i] < 32 for i in range(3)]
            near_bdd_bool = True if any(above) or any(below) else False
        return near_bdd_bool

    def init_counter(self):
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
        counter = dict()
        for label in self.labeled_target_graphs.keys():
            counter[label] = 0
        return counter

    def init_tracker(self):
        tracker = dict()
        for label in self.labeled_target_graphs.keys():
            tracker[label] = set()
        return tracker

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
        graph = self.labeled_target_graphs[swc_id].copy()
        graph, merged_cnt = gutils.delete_nodes(graph, label, return_cnt=True)
        self.labeled_target_graphs[swc_id] = graph

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
        swc_ids, results = self.generate_full_results()
        avg_results = self.generate_avg_results()

        # Reformat full results
        full_results = dict()
        for i, swc_id in enumerate(swc_ids):
            full_results[swc_id] = dict(
                [(key, results[key][i]) for key in results.keys()]
            )

        return full_results, avg_results

    def generate_full_results(self):
        """
        Generates a report by creating a list of the results for each metric.
        Each item in this list corresponds to a graph in labeled_target_graphs
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
            computing that metric for each graph in labeled_target_graphs.

        """
        swc_ids = list(self.labeled_target_graphs.keys())
        swc_ids.sort()
        stats = {
            "# splits": generate_result(swc_ids, self.split_cnts),
            "# merges": generate_result(swc_ids, self.merge_cnts),
            "% omit": generate_result(swc_ids, self.omit_percents),
            "% merged": generate_result(swc_ids, self.merged_percents),
            "edge accuracy": generate_result(swc_ids, self.edge_accuracy),
            "erl": generate_result(swc_ids, self.erl),
            "normalized erl": generate_result(swc_ids, self.normalized_erl),
        }
        return swc_ids, stats

    def generate_avg_results(self):
        avg_stats = {
            "# splits": self.avg_result(self.split_cnts),
            "# merges": self.avg_result(self.merge_cnts),
            "% omit": self.avg_result(self.omit_percents),
            "% merged": self.avg_result(self.merged_percents),
            "edge accuracy": self.avg_result(self.edge_accuracy),
            "erl": self.avg_result(self.erl),
            "normalized erl": self.avg_result(self.normalized_erl),
        }
        return avg_stats

    def avg_result(self, stats):
        result = []
        wgts = []
        for swc_id, wgt in self.wgts.items():
            if self.omit_percents[swc_id] < 1:
                result.append(stats[swc_id])
                wgts.append(wgt)
        return np.average(result, weights=wgts)

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
        self.wgts = dict()
        total_path_length = 0
        for swc_id in self.target_graphs.keys():
            pred_graph = self.labeled_target_graphs[swc_id]
            target_graph = self.target_graphs[swc_id]

            path_length = gutils.compute_path_length(target_graph)
            run_lengths = gutils.compute_run_lengths(pred_graph)
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            self.erl[swc_id] = np.sum(wgt * run_lengths)
            self.normalized_erl[swc_id] = self.erl[swc_id] / path_length

            self.wgts[swc_id] = path_length
            total_path_length += path_length

        for swc_id in self.target_graphs.keys():
            self.wgts[swc_id] = self.wgts[swc_id] / total_path_length

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
            "% omit",
            "% merged",
            "edge accuracy",
            "erl",
            "normalized erl",
        ]
        return metrics

    def save_swc(self, xyz_1, xyz_2, mistake_type):
        xyz_1 = utils.to_world(xyz_1, self.anisotropy)
        xyz_2 = utils.to_world(xyz_2, self.anisotropy)
        if mistake_type == "split":
            color = "0.0 1.0 0.0"
            cnt = 1 + np.sum(list(self.split_cnts.values())) // 2
        else:
            color = "0.0 0.0 1.0"
            cnt = 1 + np.sum(list(self.merge_cnts.values())) // 2

        path = f"{self.output_dir}/{mistake_type}-{cnt}.swc"
        save(path, xyz_1, xyz_2, color=color)


# -- utils --
def generate_result(swc_ids, stats):
    """
    Reorders items in "stats" with respect to the order defined by "swc_ids".

    Parameters
    ----------
    swc_ids : list[str]
        List of all swc_ids of graphs in "self.labeled_target_graphs".
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
