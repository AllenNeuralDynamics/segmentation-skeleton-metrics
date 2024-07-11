# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from time import time

import numpy as np
import tensorstore as ts

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import (
    merge_detection,
    split_detection,
    utils,
)
from segmentation_skeleton_metrics.swc_utils import save, to_graph

MERGE_DIST_THRESHOLD = 25


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
        target_swc_paths,
        anisotropy=[1.0, 1.0, 1.0],
        connections_path=None,
        ignore_boundary_mistakes=False,
        output_dir=None,
        valid_labels=None,
        write_sites=False,
    ):
        """
        Constructs skeleton metric object that evaluates the quality of a
        predicted segmentation.

        Parameters
        ----------
        pred_labels : numpy.ndarray or tensorstore.TensorStore
            Predicted segmentation mask.
        target_swc_paths : list[str]
            List of paths to swc files where each file corresponds to a
            neuron in the ground truth.
        anisotropy : list[float], optional
            Image to real-world coordinates scaling factors applied to swc
            files. The default is [1.0, 1.0, 1.0]
        connections_path : list[tuple]
            Path to a txt file containing pairs of swc keys from the prediction
            that were predicted to be connected.
        ignore_boundary_mistakes : bool, optional
            Indication of whether to ignore mistakes near boundary of bounding
            box. The default is False.
        output_dir : str, optional
            Path to directory that each mistake site is written to. The default
            is None.
        valid_labels : set
            ...
        write_sites, : bool, optional
            Indication of whether to write mistake sites to an swc file. The
            default is False.

        Returns
        -------
        None.

        """
        # Store options
        self.anisotropy = anisotropy
        self.ignore_boundary_mistakes = ignore_boundary_mistakes
        self.output_dir = output_dir
        self.write_sites = write_sites

        # Labels
        assert type(valid_labels) is set if valid_labels is not None else True
        self.label_mask = pred_labels
        self.valid_labels = valid_labels
        self.init_label_map(connections_path)

        # Build Graphs
        self.graphs = self.init_graphs(target_swc_paths, anisotropy)
        self.init_node_labels()

    # -- Initialize and Label Graphs --
    def init_label_map(self, path):
        if path:
            assert self.valid_labels is not None, "Must provide valid labels!"
            self.label_map = utils.init_label_map(path, self.valid_labels)
        else:
            self.label_map = None

    def init_graphs(self, paths, anisotropy):
        """
        Initializes "self.graphs" by iterating over "paths" which
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
            key = utils.get_id(path)
            graphs[key] = to_graph(path, anisotropy=anisotropy)
        return graphs

    def init_node_labels(self):
        """
        Initializes "self.graphs" by copying each graph in
        "self.graphs", then labels each node with the label in
        "self.label_mask" that coincides with it.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("\nLabeling Graphs...")
        t0 = time()
        self.key_to_label_to_nodes = dict()  # {id: {label: nodes}}
        for key, graph in self.graphs.items():
            self.key_to_label_to_nodes[key] = self.label_nodes(graph)

        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def label_nodes(self, graph):
        """
        Iterates over nodes in "graph" and stores the label in the predicted
        segmentation mask (i.e. "self.label_mask") which coincides with each
        node as a node-level attribute called "label".

        Parameters
        ----------
        graph : networkx.Graph
            Graph that represents a neuron from the ground truth.

        Returns
        -------
        networkx.Graph
            Updated graph with node-level attributes called "label".

        """
        key_to_label_to_nodes = dict()
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = []
            for i in graph.nodes:
                coord = gutils.get_coord(graph, i)
                threads.append(executor.submit(self.get_label, coord, i))

            # Store results
            for thread in as_completed(threads):
                i, label = thread.result()
                graph.nodes[i].update({"label": label})
                if label in key_to_label_to_nodes.keys():
                    key_to_label_to_nodes[label].add(i)
                else:
                    key_to_label_to_nodes[label] = set([i])
        return key_to_label_to_nodes

    def get_label(self, coord, return_node=False):
        """
        Gets label of voxel at "coord".

        Parameters
        ----------
        coord : numpy.ndarray
            Image coordinate of voxel to be read.

        Returns
        -------
        int
           Label of voxel at "coord".

        """
        label = self.read_label(coord)
        if return_node:
            return return_node, self.validate(label)
        else:
            return self.validate(label)

    def read_label(self, coord):
        """
        Gets label at image coordinates "coord".

        Parameters
        ----------
        coord : tuple[int]
            Coordinates that indexes into "self.label_mask".

        Returns
        -------
        int
           Label at image coordinates "coord".

        """
        if type(self.label_mask) == ts.TensorStore:
            return int(self.label_mask[coord].read().result())
        else:
            return self.label_mask[coord]

    def validate(self, label):
        """
        Validates label by checking whether it is contained in
        "self.valid_labels".

        Parameters
        ----------
        label : int
            Label to be validated.

        Returns
        -------
        int
            There are two possibilities: (1) original label if either "label"
            is contained in "self.valid_labels" or "self.valid_labels" is
            None, or (2) 0 if "label" is not contained in self.valid_labels.

        """
        if self.label_map:
            return self.equivalent_label(label)
        elif self.valid_labels:
            return 0 if label not in self.valid_labels else label
        else:
            return label

    def equivalent_label(self, label):
        """
        Gets the equivalence class label corresponding to "label".

        Parameters
        ----------
        label : int
            Label to be checked.

        Returns
        -------
        label
            Equivalence class label.

        """
        if label in self.label_map.keys():
            return self.label_map[label]
        else:
            return 0

    # -- Evaluation --
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
        self.saved_site_cnt = 0
        self.detect_splits()
        self.quantify_splits()

        # Merge evaluation
        print("Detecting Merges...")
        self.saved_site_cnt = 0
        self.detect_merges()
        self.quantify_merges()

        # Compute metrics
        full_results, avg_results = self.compile_results()
        return full_results, avg_results

    def get_labels(self, key):
        """
        Gets the predicted labels that intersect with the target graph
        corresponding to "id".

        Parameters
        ----------
        key : str

        Returns
        -------
        set
            Labels that intersect with the target graph corresponding to "id".
        """
        return set(self.key_to_label_to_nodes[key].keys())

    def zero_nodes(self, key, label):
        """
        Zeros out nodes in "self.graph[key]" in the sense that nodes with label
        "label" are updated to zero.

        Parameters
        ----------
        key : str
            ID of ground truth graph to be updated.
        label : int
            Label that identifies which nodes to have their label updated to
            zero.

        Returns
        -------
        None

        """
        if label in self.key_to_label_to_nodes[key].keys():
            for i in self.key_to_label_to_nodes[key][label]:
                self.graphs[key].nodes[i]["label"] = 0
            del self.key_to_label_to_nodes[key][label]

    def detect_splits(self):
        """
        Detects splits in the predicted segmentation, then deletes node and
        edges in "self.graphs" that correspond to a split.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        t0 = time()
        for cnt, (key, graph) in enumerate(self.graphs.items()):
            # Detection
            utils.progress_bar(cnt + 1, len(self.graphs))
            graph = split_detection.run(graph, self.graphs[key])

            # Update predicted graph
            self.graphs[key] = gutils.delete_nodes(graph, 0)
            self.key_to_label_to_nodes[key] = gutils.store_labels(graph)

        # Report runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def quantify_splits(self):
        """
        Counts the number of splits, number of omit edges, and percent of omit
        edges for each graph in "self.graphs".

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
        for key in self.graphs.keys():
            n_pred_edges = self.graphs[key].number_of_edges()
            n_target_edges = self.graphs[key].graph["initial_number_of_edges"]

            self.split_cnts[key] = gutils.count_splits(self.graphs[key])
            self.omit_cnts[key] = n_target_edges - n_pred_edges
            self.omit_percents[key] = 1 - n_pred_edges / n_target_edges

    def detect_merges(self):
        """
        Detects merges in the predicted segmentation, then deletes node and
        edges in "self.graphs" that correspond to a merge.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initilizations
        self.merge_cnts = self.init_counter()
        self.merged_cnts = self.init_counter()
        self.merged_percents = self.init_counter()

        # Check potential merge sites
        t0 = time()
        with ProcessPoolExecutor() as executor:
            processes = []
            for keys, label in self.detect_potential_merges():
                key_1, key_2 = tuple(keys)
                processes.append(
                    executor.submit(
                        merge_detection.localize,
                        self.graphs[key_1],
                        self.graphs[key_2],
                        self.key_to_label_to_nodes[key_1][label],
                        self.key_to_label_to_nodes[key_2][label],
                        MERGE_DIST_THRESHOLD,
                        (keys, label),
                    )
                )

            # Compile results
            cnt = 1
            chunk_size = len(processes) * 0.02
            detected_merges = set()
            for i, process in enumerate(as_completed(processes)):
                # Check site
                merge_id, site, d = process.result()
                if d < MERGE_DIST_THRESHOLD:
                    detected_merges.add(merge_id)
                    if self.write_sites:
                        self.save_merge_sites(site[0], site[1], "merge")

                # Report process
                if i >= cnt * chunk_size:
                    utils.progress_bar(i + 1, len(processes))
                    cnt += 1

        # Update graph
        for (key_1, key_2), label in detected_merges:
            self.process_merge(key_1, label)
            self.process_merge(key_2, label)

        # Report Runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def detect_potential_merges(self):
        """
        Detects merges between ground truth graphs which are considered to be
        potential merge sites.

        Parameters
        ----------
        None

        Returns
        -------
        set
            Set of tuples containing a tuple of graph keys and common label
            between the graphs.

        """
        return merge_detection.find_sites(self.graphs, self.get_labels)

    def near_bdd(self, xyz):
        """
        Determines whether "xyz" is near the boundary of the image.

        Parameters
        ----------
        xyz : numpy.ndarray
            xyz coordinate to be checked

        Returns
        -------
        bool
            Indication of whether "xyz" is near the boundary of the image.

        """
        near_bdd_bool = False
        if self.ignore_boundary_mistakes:
            mask_shape = self.label_mask.shape
            above = [xyz[i] >= mask_shape[i] - 32 for i in range(3)]
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
        for key in self.graphs.keys():
            counter[key] = 0
        return counter

    def init_tracker(self):
        tracker = dict()
        for key in self.graphs.keys():
            tracker[key] = set()
        return tracker

    def process_merge(self, key, label):
        """
        Once a merge has been detected that corresponds to "id", every
        node in "self.graphs[key]" with that "label" is
        deleted.

        Parameters
        ----------
        str
            Key associated with the graph to be searched.
        int
            Label in prediction that is assocatied with a merge.

        Returns
        -------
        None

        """
        graph = self.graphs[key].copy()
        graph, merged_cnt = gutils.delete_nodes(graph, label, return_cnt=True)
        self.graphs[key] = graph
        self.merged_cnts[key] += merged_cnt
        self.merge_cnts[key] += 1

    def quantify_merges(self):
        """
        Computes the percentage of merged edges for each graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.merged_percents = dict()
        for key in self.graphs.keys():
            n_edges = self.graphs[key].number_of_edges()
            percent = self.merged_cnts[key] / n_edges
            self.merged_percents[key] = percent

    def compile_results(self):
        """
        Compiles a dictionary containing the metrics computed by this module.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary where the keys are keys and the values are the
            result of computing each metric for the corresponding graphs.
        dict
            Dictionary where the keys are names of metrics computed by this
            module and values are the averaged result over all keys.

        """
        # Compute remaining metrics
        self.compute_edge_accuracy()
        self.compute_erl()

        # Summarize results
        keys, results = self.generate_full_results()
        avg_results = self.generate_avg_results()

        # Reformat full results
        full_results = dict()
        for i, key in enumerate(keys):
            full_results[key] = dict(
                [(key, results[key][i]) for key in results.keys()]
            )

        return full_results, avg_results

    def generate_full_results(self):
        """
        Generates a report by creating a list of the results for each metric.
        Each item in this list corresponds to a graph in self.graphs and
        this list is ordered with respect to "keys".

        Parameters
        ----------
        None

        Results
        -------
        list[str]
            Specifies the ordering of results for each value in "stats".
        dict
            Dictionary where the keys are metrics and values are the result of
            computing that metric for each graph in self.graphs.

        """
        keys = list(self.graphs.keys())
        keys.sort()
        stats = {
            "# splits": generate_result(keys, self.split_cnts),
            "# merges": generate_result(keys, self.merge_cnts),
            "% omit": generate_result(keys, self.omit_percents),
            "% merged": generate_result(keys, self.merged_percents),
            "edge accuracy": generate_result(keys, self.edge_accuracy),
            "erl": generate_result(keys, self.erl),
            "normalized erl": generate_result(keys, self.normalized_erl),
        }
        return keys, stats

    def generate_avg_results(self):
        avg_stats = {
            "# splits": self.avg_result(self.split_cnts),
            "# merges": self.avg_result(self.merge_cnts) / 2,
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
        for key, wgt in self.wgts.items():
            if self.omit_percents[key] < 1:
                result.append(stats[key])
                wgts.append(wgt)
        return np.average(result, weights=wgts)

    def compute_edge_accuracy(self):
        """
        Computes the edge accuracy of each self.graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.edge_accuracy = dict()
        for key in self.graphs.keys():
            omit_percent = self.omit_percents[key]
            merged_percent = self.merged_percents[key]
            self.edge_accuracy[key] = 1 - omit_percent - merged_percent

    def compute_erl(self):
        """
        Computes the expected run length (ERL) of each graph.

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
        total_run_length = 0
        for key in self.graphs.keys():
            run_length = self.graphs[key].graph["initial_run_length"]
            run_lengths = gutils.compute_run_lengths(self.graphs[key])
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            self.erl[key] = np.sum(wgt * run_lengths)
            self.normalized_erl[key] = self.erl[key] / run_length

            self.wgts[key] = run_length
            total_run_length += run_length

        for key in self.graphs.keys():
            self.wgts[key] = self.wgts[key] / total_run_length

    def list_metrics(self):
        """
        Lists metrics that are computed by this module.

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Metrics computed by this module.

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

    def save_merge_sites(self, xyz_1, xyz_2, mistake_type):
        self.saved_site_cnt += 1
        xyz_1 = utils.to_world(xyz_1, self.anisotropy)
        xyz_2 = utils.to_world(xyz_2, self.anisotropy)
        color = "0.0 1.0 0.0" if mistake_type == "split" else "0.0 0.0 1.0"
        path = f"{self.output_dir}/{mistake_type}-{self.saved_site_cnt}.swc"
        save(path, xyz_1, xyz_2, color=color)


# -- utils --
def generate_result(keys, stats):
    """
    Reorders items in "stats" with respect to the order defined by "keys".

    Parameters
    ----------
    keys : list[str]
        List of all "keys" of graphs in "self.graphs".
    stats : dict
        Dictionary where the keys are "keys" and values are the result
        of computing some metrics.

    Returns
    -------
    list
        Reorded items in "stats" with respect to the order defined by
        "keys".

    """
    return [stats[key] for key in keys]
