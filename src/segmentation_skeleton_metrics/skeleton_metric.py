# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from time import time
from zipfile import ZipFile

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import KDTree

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import (
    merge_detection,
    split_detection,
    swc_utils,
    utils,
)
from segmentation_skeleton_metrics.merge_detection import find_sites

MERGE_DIST_THRESHOLD = 20


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
        pred_swc_paths=None,
        valid_labels=None,
        save_projections=False,
        save_sites=False,
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
            files at "target_swc_paths". The default is [1.0, 1.0, 1.0].
        connections_path : list[tuple]
            Path to a txt file containing pairs of segment ids of segments
            that were merged into a single segment.
        ignore_boundary_mistakes : bool, optional
            Indication of whether to ignore mistakes near boundary of bounding
            box. The default is False.
        output_dir : str, optional
            Path to directory that mistake sites are written to. The default
            is None.
        pred_swc_paths : str, optional
            List of paths to swc files where each file corresponds to a
            neuron from the prediction. If provided, these fragments are used
            to compute the 'projected run length' (see merge_detection.py for
            details). The default is None.
        valid_labels : set[int], optional
            Segment ids (i.e. labels) that are present in the segmentation.
            The purpose of this argument is to account for segments that were
            removed due to thresholding by path length. The default is None.
        save_projections: bool, optional
            ...
        save_sites, : bool, optional
            Indication of whether to write merge sites to an swc file. The
            default is False.

        Returns
        -------
        None.

        """
        # Options
        self.anisotropy = anisotropy
        self.connections_path = connections_path
        self.ignore_boundary_mistakes = ignore_boundary_mistakes
        self.output_dir = output_dir
        self.pred_swc_paths = pred_swc_paths
        self.save_sites = save_sites

        # Labels and Graphs
        assert type(valid_labels) is set if valid_labels else True
        self.label_mask = pred_labels
        self.valid_labels = valid_labels
        self.init_label_map(connections_path)
        self.init_graphs(target_swc_paths, anisotropy)

        # Initialize writer
        self.save_projections = save_projections
        if self.save_projections:
            self.init_zip_writer()

    # -- Initialize and Label Graphs --
    def init_label_map(self, path):
        """
        Initializes a dictionary that maps a label to its equivalent label in
        the case where "connections_path" is provided.

        Parameters
        ----------
        path : str
            Path to a txt file containing pairs of segment ids of segments
            that were merged into a single segment.

        Returns
        -------
        None

        """
        if path:
            assert self.valid_labels is not None, "Must provide valid labels!"
            self.label_map, self.inv_label_map = utils.init_label_map(
                path, self.valid_labels
            )
        else:
            self.label_map = None
            self.inv_label_map = None

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
        reader = swc_utils.Reader(anisotropy=anisotropy, return_graphs=True)
        self.graphs = reader.load_from_local_paths(paths)
        self.init_key_label_nodes()

    def init_key_label_nodes(self):
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
        print("Labeling Graphs...")
        t0 = time()
        cnt = 0
        self.key_to_label_to_nodes = dict()  # {id: {label: nodes}}
        for key, graph in self.graphs.items():
            utils.progress_bar(cnt + 1, len(self.graphs))
            self.key_to_label_to_nodes[key] = self.set_labels(graph)
            cnt += 1

        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def set_labels(self, graph):
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
        dict
            Dictionary that maps a label to the subset of nodes in "graph"
            that have that label.

        """
        label_to_nodes = dict()
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
                if label in label_to_nodes:
                    label_to_nodes[label].add(i)
                else:
                    label_to_nodes[label] = set([i])
        return label_to_nodes

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

    def get_labels(self, key):
        """
        Gets the set of labels contained in the graph corresponding to "key".

        Parameters
        ----------
        key : str

        Returns
        -------
        set
            Labels contained in the graph corresponding to "key".
        """
        return set(self.key_to_label_to_nodes[key].keys())

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

    # -- Load Fragments --
    def load_fragments(self):
        """
        Loads and filters swc files from a local zip. These swc files are
        assumed to be fragments from a predicted segmentation. Note: Hard
        coded to read from a local zip

        Parameters
        ----------
        zip_path : str
            Path to the local zip file containing the fragments

        Returns
        -------
        dict
            Dictionary that maps an swc id to the fragment graph.

        """
        # Read fragments
        t0 = time()
        print("Loading Fragments")
        reader = swc_utils.Reader(return_graphs=True)
        fragment_graphs = reader.load_from_local_zip(self.pred_swc_paths)

        # Filter fragments
        self.graph_to_labels = gutils.get_node_labels(self.graphs)
        self.fragment_graphs = self.filter_fragments(fragment_graphs)
        print("\n# Fragments:", len(self.fragment_graphs))

        t, unit = utils.time_writer(time() - t0)
        print(f"Runtime: {round(t, 2)} {unit}\n")

    def filter_fragments(self, fragment_graphs):
        """
        Filters the provided "fragment_graphs" dictionary to include only
        those fragments whose labels are present in the combined set of node
        labels extracted from "self.graphs".

        Parameters
        ----------
        fragment_graphs : dict
            Dictionary that maps a label and to the corresponding fragment
            graph.

        Returns
        -------
        dict
            Dictionary containing only those fragment graphs whose labels are
            present in the combined set of labels from "self.graphs". The keys
            are the labels and the values are the corresponding fragment
            graphs.

        """
        labels = set.union(*list(self.graph_to_labels.values()))
        return {l: fragment_graphs[l] for l in labels if l in fragment_graphs}

    def init_fragment_arrays(self):
        """
        Initializes "self.fragment_arrays" by converting the graphs stored in
        "self.fragment_graphs" into xyz arrays by extracting the node's xyz
        coordinates.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.fragment_arrays = dict()
        for label, graph in self.fragment_graphs.items():
            self.fragment_arrays[label] = gutils.to_xyz_array(graph)
        del self.fragment_graphs

    def init_zip_writer(self):
        """
        Initializes "self.zip_writer" attribute by setting up a directory for
        output files and creating ZIP files for each graph in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initialize output directory
        output_dir = os.path.join(self.output_dir, "projections")
        utils.mkdir(output_dir)

        # Save intial graphs
        self.zip_writer = dict()
        for key in self.graphs.keys():
            self.zip_writer[key] = ZipFile(f"{output_dir}/{key}.zip", "w")
            swc_utils.to_zipped_swc(
                self.zip_writer[key], self.graphs[key], color="1.0 0.0 0.0"
            )

    # -- Main Routine --
    def run(self):
        """
        Computes skeleton-based metrics.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            ...

        """
        # Split evaluation
        print("Detecting Splits...")
        self.saved_site_cnt = 0
        self.detect_splits()
        self.quantify_splits()

        # Projected run lengths
        if self.pred_swc_paths:
            print("Computing Projected Run Lengths...")
            self.load_fragments()
            self.compute_projected_run_lengths()

        # Merge evaluation
        print("Detecting Merges...")
        self.saved_site_cnt = 0
        self.detect_merges()
        self.quantify_merges()

        # Compute metrics
        full_results, avg_results = self.compile_results()
        return full_results, avg_results

    # -- Projected Run Lengths --
    def compute_projected_run_lengths(self):
        """
        Computes the projected run length for each graph in "self.graphs".
        First, we detect fragments from "self.pred_swc_paths" that are
        sufficiently close (as determined by projection distances) to the
        given graph. The projected run length is the sum of the path lengths
        of fragments that were detected.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initializations
        self.run_length_ratio = dict()
        self.projected_run_length = dict()
        self.target_run_length = dict()

        # Compute run lengths
        t0 = time()
        for key, labels in self.graph_to_labels.items():
            target_rl = self.get_run_length(key)
            projected_rl = self.compute_projected_run_length(
                labels, self.zip_writer[key]
            )

            self.projected_run_length[key] = projected_rl
            self.target_run_length[key] = target_rl
            self.run_length_ratio[key] = projected_rl / target_rl

        # Report runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"Runtime: {round(t, 2)} {unit}\n")

    def compute_projected_run_length(self, labels, zip_writer):
        run_length = 0
        for key in labels:
            if self.inv_label_map:
                run_length += self.compute_projected_run_length_with_map(
                    key, zip_writer
                )
            elif key in self.fragment_graphs:
                rl = self.fragment_graphs[key].graph["run_length"]
                run_length += rl
                if self.save_projections:
                    swc_utils.to_zipped_swc(
                        zip_writer, self.fragment_graphs[key]
                    )
        return run_length

    def compute_projected_run_length_with_map(self, key, zip_writer):
        run_length = 0
        if key in self.inv_label_map:
            for swc_id in self.inv_label_map[key]:
                if swc_id in self.fragment_graphs:
                    run_length += self.fragment_graphs[swc_id].graph[
                        "run_length"
                    ]
                    if self.save_projections:
                        swc_utils.to_zipped_swc(
                            zip_writer, self.fragment_graphs[swc_id]
                        )
        return run_length

    # -- Split Detection --
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

            # Update graph by removing omits
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
        self.split_cnt = dict()
        self.omit_cnts = dict()
        self.omit_percent = dict()
        for key in self.graphs:
            n_pred_edges = self.graphs[key].number_of_edges()
            n_target_edges = self.graphs[key].graph["number_of_edges"]

            self.split_cnt[key] = gutils.count_splits(self.graphs[key])
            self.omit_cnts[key] = n_target_edges - n_pred_edges
            self.omit_percent[key] = 1 - n_pred_edges / n_target_edges

    # -- Merge Detection --
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
        self.merge_cnt = self.init_counter()
        self.merged_cnts = self.init_counter()
        self.merged_percent = self.init_counter()
        self.merged_labels = set()
        # add conditional that checks whether detected merges file exists
        # to adjust for scenario when evaluating a corrected segmentation

        # Merges between ground truth - percent merged
        detected_merges = []
        for (key_1, key_2), label in find_sites(self.graphs, self.get_labels):
            if self.is_valid_merge(key_1, key_2, label):
                self.process_merge(key_1, label)
                self.process_merge(key_2, label)
                detected_merges.append(((key_1, key_2), label))

        # Merges outside of ground truth - number of merges
        self.init_fragment_arrays()
        for key, graph in self.graphs.items():
            xyz_array = gutils.to_xyz_array(graph)
            kdtree = KDTree(xyz_array)
            self.count_merges(key, kdtree)

    def is_valid_merge(self, key_1, key_2, label):
        """
        Checks whether merge is valid (i.e. not a few nodes with merged
        label).

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Indication of whether merge is valid.

        """
        n_merged_nodes_1 = len(self.key_to_label_to_nodes[key_1][label])
        n_merged_nodes_2 = len(self.key_to_label_to_nodes[key_2][label])
        is_valid = n_merged_nodes_1 > 30 and n_merged_nodes_2 > 30
        return True if is_valid else False

    def count_merges(self, key, kdtree):
        for label in self.graph_to_labels[key]:
            if label in self.fragment_arrays:
                for xyz in self.fragment_arrays[label][::4]:
                    d, _ = kdtree.query(xyz, k=1)
                    if d > 40:
                        self.merge_cnt[key] += 1
                        self.merged_labels.add(label)
                        break

    def localize_merges_old(self, detected_merges):
        """
        Searches for exact site where a merge occurs.

        Parameters
        ----------
        detected_merges : list[tuple[str, str, int]]
            Merge sites indicated by a pair of keys and label.

        Returns
        -------
        None

        """
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = []
            for (key_1, key_2), label in detected_merges:
                processes.append(
                    executor.submit(
                        merge_detection.localize,
                        self.graphs[key_1],
                        self.graphs[key_2],
                        self.key_to_label_to_nodes[key_1][label],
                        self.key_to_label_to_nodes[key_2][label],
                        MERGE_DIST_THRESHOLD,
                        ((key_1, key_2), label),
                    )
                )

            # Compile results
            cnt = 1
            for i, process in enumerate(as_completed(processes)):
                # Check site
                site, d = process.result()
                if d < MERGE_DIST_THRESHOLD:
                    self.save_merge_site(site[0], site[1])

                # Report process
                if i >= cnt * len(processes) * 0.02:
                    utils.progress_bar(i + 1, len(processes))
                    cnt += 1

    def save_merge_site(self, xyz_1, xyz_2):
        """
        Saves the site where a merge is located by writing the xyz coordinates
        to an swc file.

        Parameters
        ----------
        xyz_1 : numpy.ndarray
            xyz coordinate of merge site.
        xyz_2 : numpy.ndarray
            xyz coordinate of merge site.

        Returns
        -------
        None

        """
        self.saved_site_cnt += 1
        xyz_1 = utils.to_world(xyz_1, self.anisotropy)
        xyz_2 = utils.to_world(xyz_2, self.anisotropy)
        color = "1.0 0.0 0.0"
        path = f"{self.output_dir}/merge-{self.saved_site_cnt}.swc"
        swc_utils.save(path, xyz_1, xyz_2, color=color)

    def process_merge(self, key, label):
        """
        Once a merge has been detected that corresponds to "key", every
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
        self.merged_labels.add(label)

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
        self.merged_percent = dict()
        for key in self.graphs:
            n_edges = self.graphs[key].graph["number_of_edges"]
            percent = self.merged_cnts[key] / n_edges
            self.merged_percent[key] = percent

    # -- Compute Metrics --
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
            full_results[key] = {key: val[i] for key, val in results.items()}
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
            "# splits": generate_result(keys, self.split_cnt),
            "# merges": generate_result(keys, self.merge_cnt),
            "% omit": generate_result(keys, self.omit_percent),
            "% merged": generate_result(keys, self.merged_percent),
            "edge accuracy": generate_result(keys, self.edge_accuracy),
            "projected_rl": generate_result(keys, self.projected_run_length),
            "target_rl": generate_result(keys, self.target_run_length),
            "rl_ratio": generate_result(keys, self.run_length_ratio),
            "erl": generate_result(keys, self.erl),
            "normalized erl": generate_result(keys, self.normalized_erl),
        }
        return keys, stats

    def generate_avg_results(self):
        """
        Averages value of each metric across all graphs from "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Average value of each metric across "self.graphs".

        """
        avg_stats = {
            "# splits": self.avg_result(self.split_cnt),
            "# merges": self.avg_result(self.merge_cnt),
            "% omit": self.avg_result(self.omit_percent),
            "% merged": self.avg_result(self.merged_percent),
            "edge accuracy": self.avg_result(self.edge_accuracy),
            "projected_rl": self.avg_result(self.projected_run_length),
            "target_rl": self.avg_result(self.target_run_length),
            "rl_ratio": self.avg_result(self.run_length_ratio),
            "erl": self.avg_result(self.erl),
            "normalized erl": self.avg_result(self.normalized_erl),
        }
        return avg_stats

    def avg_result(self, stats):
        """
        Averages the values computed across "self.graphs" for
        a given metric stored in "stats".

        Parameters
        ----------
        stats : dict
            Values computed across all graphs from "self.graphs" for a given
         metric stored in "stats".

        Returns
        -------
        float
            Average value of metric computed across self.graphs".

        """
        result = []
        wgts = []
        for key, wgt in self.wgts.items():
            if self.omit_percent[key] < 1:
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
            omit_percent = self.omit_percent[key]
            merged_percent = self.merged_percent[key]
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
            run_length = self.get_run_length(key)
            run_lengths = gutils.compute_run_lengths(self.graphs[key])
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            self.erl[key] = np.sum(wgt * run_lengths)
            self.normalized_erl[key] = self.erl[key] / run_length

            self.wgts[key] = run_length
            total_run_length += run_length

        for key in self.graphs.keys():
            self.wgts[key] = self.wgts[key] / total_run_length

    def get_run_length(self, key):
        """
        Gets the path length of "self.graphs[key]".

        Parameters
        ----------
        key : str
            Identifier of graph of interest.

        Returns
        -------
        float
            Run length of "self.graphs[key]".

        """
        return self.graphs[key].graph["run_length"]

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

    # -- Utils --
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
        Initializes a dictionary that is used to count some type of mistake
        for each graph in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary used to count some type of mistake for each graph.

        """
        counter = dict()
        for key in self.graphs.keys():
            counter[key] = 0
        return counter

    def init_tracker(self):
        """
        Initializes a dictionary whose keys are "self.graphs.keys()" and
        values are an emtpy set.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dicionary whose keys are "self.graphs.keys()" and values are an
            emtpy set.

        """
        return {key: set() for key in self.graphs.keys()}


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
