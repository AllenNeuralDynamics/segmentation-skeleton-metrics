"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from scipy.spatial import distance, KDTree
from tqdm import tqdm
from zipfile import ZipFile

import networkx as nx
import numpy as np
import os

from segmentation_skeleton_metrics import split_detection
from segmentation_skeleton_metrics.utils import (
    graph_util as gutil,
    img_util,
    swc_util,
    util
)

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
        anisotropy=(1.0, 1.0, 1.0),
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
        gt_pointer : Any
            Pointer to ground truth swcs, see "swc_util.Reader" for further
            documentation. Note these SWC files are assumed to be stored in
            image coordinates.
        pred_labels : ArrayLike
            Predicted segmentation mask.
        anisotropy : Tuple[float], optional
            Image to physical coordinate scaling factors applied to SWC files
            stored at "fragments_pointer". The default is (1.0, 1.0, 1.0).
        connections_path : str, optional
            Path to a txt file containing pairs of segment ids of segments
            that were merged into a single segment. The default is None.
        fragments_pointer : Any, optional
            Pointer to SWC files corresponding to "pred_labels", see
            "swc_util.Reader" for further documentation. Note that these SWC
            file may be stored in physical coordiantes, but the anisotropy
            scaling factors must be provided. The default is None.
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
        # Instance attributes
        self.anisotropy = anisotropy
        self.connections_path = connections_path
        self.output_dir = output_dir
        self.preexisting_merges = preexisting_merges
        self.save_projections = save_projections

        # Label handler
        self.label_handler = gutil.LabelHandler(
            connections_path=connections_path, valid_labels=valid_labels
        )

        # Load data
        self.label_mask = pred_labels
        self.load_groundtruth(gt_pointer)
        self.load_fragments(fragments_pointer)

        # Initialize writer
        if self.save_projections:
            self.init_zip_writer()

    # --- Load Data ---
    def load_groundtruth(self, swc_pointer):
        """
        Initializes "self.graphs" by iterating over "paths" which corresponds
        to neurons in the ground truth.

        Parameters
        ----------
        paths : List[str]
            List of paths to swc files which correspond to neurons in the
            ground truth.

        Returns
        -------
        None

        """
        # Build graphs
        print("\n(1) Load Ground Truth")
        graph_builder = gutil.GraphBuilder(
            anisotropy=self.anisotropy,
            label_mask=self.label_mask,
            use_anisotropy=False,
        )
        self.graphs = graph_builder.run(swc_pointer)

        # Label nodes
        for key in tqdm(self.graphs, desc="Labeling Graphs"):
            self.label_graphs(key)

    def load_fragments(self, swc_pointer):
        print("\n(2) Load Fragments")
        if swc_pointer:
            coords_only = False  #not self.save_projections
            graph_builder = gutil.GraphBuilder(
                anisotropy=self.anisotropy,
                coords_only=coords_only,
                selected_ids=self.get_all_node_labels(),
                use_anisotropy=True,
            )
            self.fragment_graphs = graph_builder.run(swc_pointer)
            self.set_fragment_ids()
        else:
            self.fragment_graphs = None

    def set_fragment_ids(self):
        self.fragment_ids = set()
        for key in self.fragment_graphs:
            self.fragment_ids.add(util.get_segment_id(key))

    def label_graphs(self, key, batch_size=64):
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
            batch = set()
            threads = list()
            visited = set()
            for i, j in nx.dfs_edges(self.graphs[key]):
                # Check for new batch
                if len(batch) == 0:
                    root = i
                    batch.add(i)
                    visited.add(i)

                # Check whether to submit batch
                is_node_far = self.graphs[key].dist(root, j) > 128
                is_batch_full = len(batch) >= batch_size
                if is_node_far or is_batch_full:
                    threads.append(
                        executor.submit(self.get_patch_labels, key, batch)
                    )
                    batch = set()

                # Visit j
                if j not in visited:
                    batch.add(j)
                    visited.add(j)
                    if len(batch) == 1:
                        root = j

            # Submit last thread
            threads.append(executor.submit(self.get_patch_labels, key, batch))

            # Process results
            self.graphs[key].set_labels()
            for thread in as_completed(threads):
                node_to_label = thread.result()
                for i, label in node_to_label.items():
                    self.graphs[key].labels[i] = label

    def get_patch_labels(self, key, nodes):
        bbox = self.graphs[key].get_bbox(nodes)
        label_patch = self.label_mask.read_with_bbox(bbox)
        node_to_label = dict()
        for i in nodes:
            voxel = self.to_local_voxels(key, i, bbox["min"])
            label = self.label_handler.get(label_patch[voxel])
            node_to_label[i] = label
        return node_to_label

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
        inverse_bool = self.label_handler.use_mapping()
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
            for l in self.graphs[key].get_labels():
                output = output.union(self.label_handler.inverse_mapping[l])
            return output
        else:
            return self.graphs[key].get_labels()

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
        util.mkdir(output_dir)

        # Save intial graphs
        self.zip_writer = dict()
        for key in self.graphs.keys():
            self.zip_writer[key] = ZipFile(f"{output_dir}/{key}.zip", "w")
            swc_util.to_zipped_swc(
                self.zip_writer[key], self.graphs[key],
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
        print("\n(3) Evaluation")

        # Split evaluation
        self.detect_splits()
        self.quantify_splits()

        # Check for prexisting merges
        if self.preexisting_merges:
            for key in self.graphs:
                self.adjust_metrics(key)

        # Merge evaluation
        self.detect_merges()
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
            if label in self.graphs[key].get_labels():
                # Extract subgraph
                nodes = self.graphs[key].nodes_with_label(label)
                subgraph = self.graphs[key].subgraph(nodes)

                # Adjust metrics
                n_edges = subgraph.number_of_edges()
                rls = gutil.compute_run_lengths(subgraph)
                self.graphs[key].graph["run_length"] -= np.sum(rls)
                self.graphs[key].graph["n_edges"] -= n_edges

                # Update graph
                self.graphs[key].remove_nodes_from(nodes)

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
        pbar = tqdm(total=len(self.graphs), desc="Split Detection")
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = list()
            for key, graph in self.graphs.items():
                processes.append(
                       executor.submit(
                           split_detection.run,
                           key,
                           graph,
                       )
                )

            # Store results
            self.split_percent = dict()
            for process in as_completed(processes):
                key, graph, split_percent = process.result()
                self.graphs[key] = graph
                self.split_percent[key] = split_percent
                pbar.update(1)

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
            # Get counts
            n_pred_edges = self.graphs[key].number_of_edges()
            n_target_edges = self.graphs[key].graph["n_edges"]

            # Compute stats
            self.split_cnt[key] = gutil.count_splits(self.graphs[key])
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

        # Count total merges
        if self.fragment_graphs:
            pbar = tqdm(total=len(self.graphs), desc="Merge Detection")
            for key, graph in self.graphs.items():
                if graph.number_of_nodes() > 0:
                    kdtree = KDTree(graph.voxels)
                    self.count_merges(key, kdtree)
                pbar.update(1)

        # Process merges
        for (key_1, key_2), label in self.find_label_intersections():
            self.process_merge(key_1, label, -1)
            self.process_merge(key_2, label, -1)

        for key, label, xyz in self.merged_labels:
            self.process_merge(key, label, xyz, update_merged_labels=False)

        # Write merges to local machine
        self.save_merged_labels()

    def count_merges(self, key, kdtree):
        """
        Counts the number of label merges for a given graph key based on
        whether the fragment graph corresponding to a label has a node that is
        more that 200ums away from the nearest point in
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
        for label in self.get_node_labels(key):
            nodes = self.graphs[key].nodes_with_label(label)
            if len(nodes) > MIN_CNT:
                for label in self.label_handler.get_class(label):
                    if label in self.fragment_ids:
                        self.is_fragment_merge(key, label, kdtree)

    def is_fragment_merge(self, key, label, kdtree):
        """
        Determines whether fragment corresponding to "label" is falsely merged
        to graph corresponding to "key". A fragment is said to be merged if
        there is a node in the fragment more than 200ums away from the nearest
        point in "kdtree".

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
        fragment_graph = self.find_graph_from_label(label)
        for voxel in fragment_graph.voxels:
            gt_voxel = util.kdtree_query(kdtree, voxel)
            if self.physical_dist(gt_voxel, voxel) > 150:
                # Log merge mistake
                equiv_label = self.label_handler.get(label)
                xyz = img_util.to_physical(voxel, self.anisotropy)
                self.merge_cnt[key] += 1
                self.merged_labels.add((key, equiv_label, tuple(xyz)))

                # Save merged fragment (if applicable)
                if self.save_projections and label in self.fragment_graphs:
                    swc_util.to_zipped_swc(
                        self.zip_writer[key], self.fragment_graphs[label]
                    )
                break

    def find_graph_from_label(self, label):
        for key in self.fragment_graphs:
            if label == util.get_segment_id(key):
                return self.fragment_graphs[key]

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
                    labels_1 = set(self.graphs[key_1].get_labels())
                    labels_2 = set(self.graphs[key_2].get_labels())
                    for label in labels_1.intersection(labels_2):
                        label_intersections.add((keys, label))
        return label_intersections

    def process_merge(self, key, label, xyz, update_merged_labels=True):
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
        if label in self.graphs[key].get_labels():
            # Compute metrics
            nodes = self.graphs[key].nodes_with_label(label)
            subgraph = self.graphs[key].subgraph(nodes)
            self.merged_edges_cnt[key] += subgraph.number_of_edges()

            # Update self
            self.graphs[key].remove_nodes_from(nodes)
            if update_merged_labels:
                self.merged_labels.add((key, label, -1))

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

    def save_merged_labels(self):
        """
        Saves merged labels and their corresponding coordinates to a text
        file.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Save detected merges
        prefix = "corrected_" if self.connections_path else ""
        filename = f"merged_ids-{prefix}segmentation.txt"
        with open(os.path.join(self.output_dir, filename), "w") as f:
            f.write(f" Label   -   xyz\n")
            for _, label, xyz in self.merged_labels:
                if self.label_handler.use_mapping():
                    label = self.get_merged_label(label)
                f.write(f" {label}   -   {xyz}\n")

    def get_merged_label(self, label):
        """
        Retrieves the label present in the corrected fragments that
        corresponds to the given label. Note: the given and retrieved label
        may differ in the case when two fragments are merged.

        Parameters
        ----------
        label : str
            The label for which to find the corresponding label present in the
            corrected fragments.

        Returns:
        -------
        str or list
            The first matching label found in "self.fragment_graphs.keys()" or
            the original associated labels from "inverse_label_map" if no
            matches are found.

        """
        for l in self.label_handler.get_class(label):
            if l in self.fragment_graphs.keys():
                return l
        return self.inverse_label_map[label]

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
            "% split": generate_result(keys, self.split_percent),
            "% merged": generate_result(keys, self.merged_percent),
            "edge accuracy": generate_result(keys, self.edge_accuracy),
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
            "% split": self.avg_result(self.split_percent),
            "% merged": self.avg_result(self.merged_percent),
            "edge accuracy": self.avg_result(self.edge_accuracy),
            "erl": self.avg_result(self.erl),
            "normalized erl": self.avg_result(self.normalized_erl),
        }
        return avg_stats

    def avg_result(self, stats):
        """
        Averages the values computed across "self.graphs" for a given metric
        stored in "stats".

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
            run_length = self.graphs[key].run_length
            total_run_length += run_length
            run_lengths = self.graphs[key].run_lengths()
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            self.erl[key] = np.sum(wgt * run_lengths)
            self.normalized_erl[key] = self.erl[key] / run_length
            self.wgts[key] = run_length

        for key in self.graphs:
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

    # -- util --
    def physical_dist(self, voxel_1, voxel_2):
        xyz_1 = img_util.to_physical(voxel_1, self.anisotropy)
        xyz_2 = img_util.to_physical(voxel_2, self.anisotropy)
        return distance.euclidean(xyz_1, xyz_2)

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

    def to_local_voxels(self, key, i, offset):
        voxel = np.array(self.graphs[key].voxels[i])
        offset = np.array(offset)
        return tuple(voxel - offset)


# -- util --
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
