# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from time import time
from zipfile import ZipFile

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import KDTree
from tqdm import tqdm

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import split_detection, swc_utils, utils
from segmentation_skeleton_metrics.graph_utils import to_xyz_array

ANISOTROPY = [0.748, 0.748, 1.0]
MERGE_DIST_THRESHOLD = 200
MIN_CNT = 40


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
        gt_pointer,
        pred_labels,
        anisotropy=[1.0, 1.0, 1.0],
        connections_path=None,
        fragments_pointer=None,
        output_dir=None,
        preexisting_merges=None,
        save_projections=False,
        valid_labels=None,
    ):
        """
        Constructs skeleton metric object that evaluates the quality of a
        predicted segmentation.

        Parameters
        ----------
        gt_pointer : container
            Pointer to ground truth swcs, see "swc_utils.Reader" for further
            documentation.
        pred_labels : numpy.ndarray or tensorstore.TensorStore
            Predicted segmentation mask.
        anisotropy : list[float], optional
            Image to real-world coordinates scaling factors applied to swc
            files at "gt_pointer". The default is [1.0, 1.0, 1.0].
        connections_path : str, optional
            Path to a txt file containing pairs of segment ids of segments
            that were merged into a single segment. The default is None.
        fragments_pointer : container, optional
            Pointer to fragments (i.e. swcs) corresponding to "pred_labels",
            see "swc_utils.Reader" for further documentation. The default is
            None.
        output_dir : str, optional
            Path to directory that mistake sites are written to. The default
            is None.
        preexisting_merges : list[int], optional
            List of segment IDs that are known to be create a false merge. The
            default is None.
        save_projections: bool, optional
            Indication of whether to save fragments that 'project' onto the
            ground truth neurons (i.e. there exists a node in a graph from
            "self.graphs" that is labeled with a given fragment id. The
            default is None.
        valid_labels : set[int], optional
            Segment ids (i.e. labels) that are present in the segmentation.
            The purpose of this argument is to account for segments that were
            removed due to thresholding by path length. The default is None.

        Returns
        -------
        None.

        """
        # Options
        self.anisotropy = anisotropy
        self.connections_path = connections_path
        self.output_dir = output_dir
        self.fragments_pointer = fragments_pointer
        self.preexisting_merges = preexisting_merges

        # Load Labels, Graphs, Fragments
        print("\n(1) Initializations")
        assert type(valid_labels) is set if valid_labels else True
        self.label_mask = pred_labels
        self.valid_labels = valid_labels
        self.init_label_map(connections_path)
        self.init_graphs(gt_pointer, anisotropy)
        if self.fragments_pointer:
            self.load_fragments()

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
            self.label_map, self.inverse_label_map = utils.init_label_map(
                path, self.valid_labels
            )
        else:
            self.label_map = None
            self.inverse_label_map = None

    def init_graphs(self, paths, anisotropy):
        """
        Initializes "self.graphs" by iterating over "paths" which corresponds
        to neurons in the ground truth.

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
        # Read graphs
        reader = swc_utils.Reader(anisotropy=anisotropy, return_graphs=True)
        self.graphs = reader.load(paths)
        self.fragment_graphs = None

        # Label nodes
        self.key_to_label_to_nodes = dict()  # {id: {label: nodes}}
        for key in tqdm(self.graphs, desc="Labeling Graphs"):
            self.set_node_labels(key)
            self.key_to_label_to_nodes[key] = gutils.init_label_to_nodes(
                self.graphs[key]
            )

    def set_node_labels(self, key):
        """
        Iterates over nodes in "graph" and stores the corresponding label from
        predicted segmentation mask (i.e. "self.label_mask") as a node-level
        attribute called "label".

        Parameters
        ----------
        graph : networkx.Graph
            Graph that represents a neuron from the ground truth.

        Returns
        -------
        None

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = []
            for i in self.graphs[key].nodes:
                voxel = tuple(self.graphs[key].nodes[i]["xyz"])
                threads.append(executor.submit(self.get_label, i, voxel))

            # Store label
            for thread in as_completed(threads):
                i, label = thread.result()
                self.graphs[key].nodes[i].update({"label": label})

    def get_label(self, i, voxel):
        """
        Gets label of voxel in "self.label_mask".

        Parameters
        ----------
        i : int
            Node ID.
        voxel : numpy.ndarray
            Image coordinate of voxel to be read.

        Returns
        -------
        int
           Label of voxel.

        """
        # Read label
        if type(self.label_mask) is ts.TensorStore:
            label = int(self.label_mask[voxel].read().result())
        else:
            label = self.label_mask[voxel]

        # Check whether to update label
        if self.label_map:
            label = self.get_equivalent_label(label)
        elif self.valid_labels:
            label = 0 if label not in self.valid_labels else label
        return i, label

    def get_equivalent_label(self, label):
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
        return self.label_map[label] if label in self.label_map else 0

    def get_all_node_labels(self):
        """
        Gets the a set of all unique labels from all graphs in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        set
            Set containing all unique labels from all graphs.

        """
        all_labels = set()
        inverse_bool = True if self.inverse_label_map else False
        for key in self.graphs:
            labels = self.get_node_labels(key, inverse_bool=inverse_bool)
            all_labels = all_labels.union(labels)
        return all_labels

    def get_node_labels(self, key, inverse_bool=False):
        """
        Gets the set of labels of nodes in the graph corresponding to "key".

        Parameters
        ----------
        key : str
            ID of graph in "self.graphs".
        inverse_bool : bool
            Indication of whether to return original labels from
            "self.labels_mask" in the case where labels were remapped. The
            default is False.

        Returns
        -------
        set
            Labels contained in the graph corresponding to "key".

        """
        if inverse_bool:
            output = set()
            for l in self.key_to_label_to_nodes[key].keys():
                output = output.union(self.inverse_label_map[l])
            return output
        else:
            return set(self.key_to_label_to_nodes[key].keys())

    # -- Load Fragments --
    def load_fragments(self):
        """
        Loads and filters swc files from a local zip. These swc files are
        assumed to be fragments from a predicted segmentation.

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
        anisotropy = [1.0 / a_i for a_i in ANISOTROPY]  # hard coded
        reader = swc_utils.Reader(anisotropy=anisotropy, return_graphs=True)
        fragment_graphs = reader.load(self.fragments_pointer)

        # Filter fragments
        self.fragment_graphs = dict()
        for label in self.get_all_node_labels():
            if label in fragment_graphs:
                self.fragment_graphs[label] = fragment_graphs[label]
            else:
                self.fragment_graphs[label] = nx.Graph(
                    filename=f"{label}.swc", run_length=0, n_edges=1
                )
        print("# Fragments:", len(self.fragment_graphs))

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
        print("\n(2) Evaluation")

        # Split evaluation
        self.detect_splits()
        self.quantify_splits()

        # Check whether to delete prexisting merges
        if self.preexisting_merges:
            for key in self.graphs:
                self.adjust_metrics(key)

        # Merge evaluation
        self.detect_merges()
        self.compute_projected_run_lengths()
        self.quantify_merges()

        # Compute metrics
        full_results, avg_results = self.compile_results()
        return full_results, avg_results

    def adjust_metrics(self, key):
        """
        Adjusts the metrics of the graph associated with the given key by
        removing nodes corresponding to known merges and their corresponding
        subgraphs. Updates the total number of edges and run lengths in the
        graph.

        Parameters
        ----------
        key : str
            Identifier for the graph to adjust.

        Returns
        -------
        None

        """
        for label in self.preexisting_merges:
            label = self.label_map[label] if self.label_map else label
            if label in self.key_to_label_to_nodes[key].keys():
                # Extract subgraph
                nodes = deepcopy(self.key_to_label_to_nodes[key][label])
                subgraph = self.graphs[key].subgraph(nodes)

                # Adjust metrics
                n_edges = subgraph.number_of_edges()
                rls = gutils.compute_run_lengths(subgraph)
                self.graphs[key].graph["run_length"] -= np.sum(rls)
                self.graphs[key].graph["n_edges"] -= n_edges

                # Update graph
                self.graphs[key].remove_nodes_from(nodes)
                del self.key_to_label_to_nodes[key][label]

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
        for key, graph in tqdm(self.graphs.items(), desc="Split Detection:"):
            # Detection
            graph = split_detection.run(graph, self.graphs[key])

            # Update graph by removing omits (i.e. nodes labeled 0)
            self.graphs[key] = gutils.delete_nodes(graph, 0)
            self.key_to_label_to_nodes[key] = gutils.init_label_to_nodes(
                self.graphs[key]
            )

        # Report runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"Runtime: {round(t, 2)} {unit}\n")

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
            n_target_edges = self.graphs[key].graph["n_edges"]

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
        self.merged_edges_cnt = self.init_counter()
        self.merged_percent = self.init_counter()
        self.merged_labels = set()
        self.projected_run_length = dict()

        # Count total merges
        if self.fragment_graphs:
            for key, graph in self.graphs.items():
                kdtree = KDTree(gutils.to_xyz_array(graph))
                self.count_merges(key, kdtree)

        # Process merges
        for (key_1, key_2), label in self.find_label_intersections():
            self.process_merge(key_1, label)
            self.process_merge(key_2, label)

        for key, label in self.merged_labels:
            self.process_merge(key, label, update_merged_labels=False)

    def count_merges(self, key, kdtree):
        """
        Counts the number of label merges for a given graph key based on
        whether the fragment graph corresponding to a label has a node that is
        more that MERGE_DIST_THRESHOLD-ums away from the nearest point in
        "kdtree".

        Parameters
        ----------
        key : str
            ID of graph in "self.graphs".
        kdtree : scipy.spatial.KDTree
            A KD-tree built from xyz coordinates in "self.graphs[key]".

        Returns
        -------
        None

        """
        self.projected_run_length[key] = 0
        for label in self.get_node_labels(key):
            if len(self.key_to_label_to_nodes[key][label]) > MIN_CNT:
                # Check whether to compute label inverse
                if self.inverse_label_map:
                    labels = deepcopy(self.inverse_label_map[label])
                else:
                    labels = [label]

                # Check if fragment is a merge mistake
                for label in labels:
                    rl = self.fragment_graphs[label].graph["run_length"]
                    self.projected_run_length[key] += rl
                    self.is_fragment_merge(key, label, kdtree)

    def is_fragment_merge(self, key, label, kdtree):
        """
        Determines whether fragment corresponding to "label" is falsely merged
        to graph corresponding to "key". A fragment is said to be merged if
        there is a node in the fragment more than MERGE_DIST_THRESHOLD-ums
        away from the nearest point in "kdtree".

        Parameters
        ----------
        key : str
            ID of graph in "self.graphs".
        label : int
            ID of fragment.
        kdtree : scipy.spatial.KDTree
            A KD-tree built from xyz coordinates in "self.graphs[key]".

        Returns
        -------
        None

        """
        for xyz in to_xyz_array(self.fragment_graphs[label])[::5]:
            if kdtree.query(xyz)[0] > MERGE_DIST_THRESHOLD:
                # Check whether to take inverse of label
                if self.inverse_label_map:
                    equivalent_label = self.label_map[label]
                else:
                    equivalent_label = label

                # Record merge mistake
                self.merge_cnt[key] += 1
                self.merged_labels.add((key, equivalent_label))
                if self.save_projections:
                    swc_utils.to_zipped_swc(
                        self.zip_writer[key], self.fragment_graphs[label]
                    )
                return

    def find_label_intersections(self):
        """
        Detects merges between ground truth graphs, namely distinct graphs that
        contain nodes with the same label.

        Parameters
        ----------
        None

        Returns
        -------
        set[tuple]
            Set of tuples containing a tuple of graph ids and common label
            between the graphs.

        """
        label_intersections = set()
        visited = set()
        for key_1 in self.graphs:
            for key_2 in self.graphs:
                keys = frozenset((key_1, key_2))
                if key_1 != key_2 and keys not in visited:
                    visited.add(keys)
                    labels_1 = self.get_node_labels(key_1)
                    labels_2 = self.get_node_labels(key_2)
                    for label in labels_1.intersection(labels_2):
                        label_intersections.add((keys, label))
        return label_intersections

    def process_merge(self, key, label, update_merged_labels=True):
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
        if label in self.key_to_label_to_nodes[key]:
            # Compute metrics
            nodes = list(self.key_to_label_to_nodes[key][label])
            subgraph = self.graphs[key].subgraph(nodes)
            self.merged_edges_cnt[key] += subgraph.number_of_edges()

            # Update self
            self.graphs[key].remove_nodes_from(nodes)
            del self.key_to_label_to_nodes[key][label]
            if update_merged_labels:
                self.merged_labels.add((key, label))

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
            n_edges = self.graphs[key].graph["n_edges"]
            self.merged_percent[key] = self.merged_edges_cnt[key] / n_edges

    # -- Projected Run Lengths --
    def compute_projected_run_lengths(self):
        """
        Computes the projected run length for each graph in "self.graphs".
        First, we detect fragments from "self.fragments_pointer" that are
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
        self.target_run_length = dict()

        # Compute run lengths
        for key in self.graphs:
            target_rl = self.get_run_length(key)
            projected_rl = self.projected_run_length[key]

            self.projected_run_length[key] = projected_rl
            self.target_run_length[key] = target_rl
            self.run_length_ratio[key] = projected_rl / target_rl

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
        for key in self.graphs:
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
        for key in self.graphs:
            run_length = self.get_run_length(key)
            run_lengths = gutils.compute_run_lengths(self.graphs[key])
            total_run_length += run_length
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            self.erl[key] = np.sum(wgt * run_lengths)
            self.normalized_erl[key] = self.erl[key] / run_length
            self.wgts[key] = run_length

        for key in self.graphs:
            self.wgts[key] = self.wgts[key] / total_run_length

    def get_run_length(self, key):
        """
        Gets the path length of "self.graphs[key]".

        Parameters
        ----------
        key : str
            ID of graph in "self.graphs".

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
        return {key: 0 for key in self.graphs}


# -- utils --
def find_sites(graphs, get_labels):
    """
    Detects merges between ground truth graphs which are considered to be
    potential merge sites.

    Parameters
    ----------
    graphs : dict
        Dictionary where the keys are graph ids and values are graphs.
    get_labels : func
        Gets the label of a node in "graphs".

    Returns
    -------
    merge_ids : set[tuple]
        Set of tuples containing a tuple of graph ids and common label between
        the graphs.

    """
    merge_ids = set()
    visited = set()
    for key_1 in graphs:
        for key_2 in graphs:
            keys = frozenset((key_1, key_2))
            if key_1 != key_2 and keys not in visited:
                visited.add(keys)
                intersection = get_labels(key_1).intersection(
                    get_labels(key_2)
                )
                for label in intersection:
                    merge_ids.add((keys, label))
    return merge_ids


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
