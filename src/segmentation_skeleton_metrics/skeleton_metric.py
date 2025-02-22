"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of class that computes skeleton-based metrics by comparing a
predicted neuron segmentation to a set of ground truth graphs.

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
    util,
)


class SkeletonMetric:
    """
    Class that evaluates the quality of a predicted segmentation by comparing
    the ground truth skeletons to the predicted segmentation mask. The
    accuracy is then quantified by detecting splits and merges, then computing
    the following metrics:
        (1) # Splits
        (2) # Merges
        (3) Omit Edge Ratio
        (4) Split Edge Ratio
        (5) Merged Edge Ratio
        (6) Edge accuracy
        (7) Expected Run Length (ERL)
        (8) Normalized ERL

    Class attributes
    ----------------
    merge_dist : float
        ...
    min_label_cnt : int
        ...

    """

    def __init__(
        self,
        gt_pointer,
        label_mask,
        anisotropy=(1.0, 1.0, 1.0),
        connections_path=None,
        fragments_pointer=None,
        output_dir=None,
        preexisting_merges=None,
        save_merges=False,
        valid_labels=None,
    ):
        """
        Instantiates a SkeletonMetric object that evaluates the topological
        accuracy of a predicted segmentation.

        Parameters
        ----------
        gt_pointer : Any
            Pointer to ground truth SWC files, see "swc_util.Reader" for
            documentation. These SWC files are assumed to be stored in voxel
            coordinates.
        label_mask : ImageReader
            Predicted segmentation mask.
        anisotropy : Tuple[float], optional
            Image to physical coordinate scaling factors applied to SWC files
            stored at "fragments_pointer". The default is (1.0, 1.0, 1.0).
        connections_path : str, optional
            Path to a txt file containing pairs of segment IDs that represents
            fragments that were merged. The default is None.
        fragments_pointer : Any, optional
            Pointer to SWC files corresponding to "label_mask", see
            "swc_util.Reader" for documentation. Notes: (1) "anisotropy" is
            applied to these SWC files and (2) these SWC files are required
            for counting merges. The default is None.
        output_dir : str, optional
            Path to directory wehere results are written. The default is None.
        preexisting_merges : List[int], optional
            List of segment IDs that are known to contain a merge mistake. The
            default is None.
        save_merges: bool, optional
            Indication of whether to save fragments with a merge mistake. The
            default is None.
        valid_labels : set[int], optional
            Segment IDs that can be assigned to nodes. This argument accounts
            for segments that were been removed due to some type of filtering.
            The default is None.

        Returns
        -------
        None

        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.connections_path = connections_path
        self.output_dir = output_dir
        self.preexisting_merges = preexisting_merges
        self.save_merges = save_merges

        # Label handler
        self.label_handler = gutil.LabelHandler(
            connections_path=connections_path, valid_labels=valid_labels
        )

        # Load data
        self.label_mask = label_mask
        self.load_groundtruth(gt_pointer)
        self.load_fragments(fragments_pointer)

        # Initialize writer
        if self.save_merges:
            self.init_zip_writer()

    # --- Load Data ---
    def load_groundtruth(self, swc_pointer):
        """
        Loads ground truth graphs and initializes the "graphs" attribute.

        Parameters
        ----------
        swc_pointer : Any
            Pointer to ground truth SWC files.

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
        """
        Loads fragments generated from the segmentation and initializes the
        "fragment_graphs" attribute.

        Parameters
        ----------
        swc_pointer : Any
            Pointer to predicted SWC files if provided.

        Returns
        -------
        None

        """
        print("\n(2) Load Fragments")
        if swc_pointer:
            graph_builder = gutil.GraphBuilder(
                anisotropy=self.anisotropy,
                selected_ids=self.get_all_node_labels(),
                use_anisotropy=True,
            )
            self.fragment_graphs = graph_builder.run(swc_pointer)
            self.set_fragment_ids()
        else:
            self.fragment_graphs = None

    def set_fragment_ids(self):
        """
        Sets the "fragment_ids" attribute by extracting unique segment IDs
        from the "fragment_graphs" keys.

        Returns
        -------
        None

        """
        self.fragment_ids = set()
        for key in self.fragment_graphs:
            self.fragment_ids.add(util.get_segment_id(key))

    def label_graphs(self, key):
        """
        Iterates over nodes in "graph" and stores the corresponding label from
        "self.label_mask") as a node-level attribute called "labels".

        Parameters
        ----------
        key : str
            Unique identifier of graph to be labeled.

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
                # Check if starting new batch
                if len(batch) == 0:
                    root = i
                    batch.add(i)
                    visited.add(i)

                # Check whether to submit batch
                is_node_far = self.graphs[key].dist(root, j) > 128
                is_batch_full = len(batch) >= 128
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

            # Submit last batch
            threads.append(executor.submit(self.get_patch_labels, key, batch))

            # Store results
            self.graphs[key].init_labels()
            for thread in as_completed(threads):
                node_to_label = thread.result()
                for i, label in node_to_label.items():
                    self.graphs[key].labels[i] = label

    def get_patch_labels(self, key, nodes):
        """
        Gets the labels for a given set of nodes within a specified patch of
        the label mask.

        Parameters
        ----------
        key : str
            Unique identifier of graph to be labeled.
        nodes : list
            A list of node IDs for which the labels are to be retrieved.

        Returns
        -------
        dict
            A dictionary mapping node IDs to their respective labels.
        """
        bbox = self.graphs[key].get_bbox(nodes)
        label_patch = self.label_mask.read_with_bbox(bbox)
        node_to_label = dict()
        for i in nodes:
            voxel = self.to_local_voxels(key, i, bbox["min"])
            label = self.label_handler.get(label_patch[voxel])
            node_to_label[i] = label
        return node_to_label

    # --------- HERE
    def get_all_node_labels(self):
        """
        Gets the a set of unique labels from all graphs in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        Set[int]
            Set containing unique labels from all graphs.

        """
        all_labels = set()
        inverse_bool = self.label_handler.use_mapping()
        for key in self.graphs:
            labels = self.get_node_labels(key, inverse_bool=inverse_bool)
            all_labels = all_labels.union(labels)
        return all_labels

    def get_node_labels(self, key, inverse_bool=False):
        """
        Gets the set of labels for nodes in the graph corresponding to the
        given key.

        Parameters
        ----------
        key : str
            Unique identifier of graph from which to retrieve the node labels.
        inverse_bool : bool
            Indication of whether to return the labels (from "labels_mask") or
            a remapping of these labels in the case when "connections_path" is
            provided. The default is False.

        Returns
        -------
        Set[int]
            Labels corresponding to nodes in the graph identified by "key".

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
            self.graphs[key].to_zipped_swc(self.zip_writer[key])

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

        # Merge evaluation
        self.detect_merges()
        self.quantify_merges()

        # Compute metrics
        return self.compile_results()

    # -- Split Detection --
    def detect_splits(self):
        """
        Detects split and omit edges in the labeled ground truth graphs, then
        removes omit nodes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pbar = tqdm(total=len(self.graphs), desc="Split Detection")
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Assign processes
            processes = list()
            for key, graph in self.graphs.items():
                processes.append(
                    executor.submit(split_detection.run, key, graph)
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
        Counts the number of splits, number of omit edges, and omit edge ratio
        in the labeled ground truth graphs.

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
        --> HERE

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

        # Detect merges by comparing fragment graphs to ground truth graphs
        if self.fragment_graphs:
            pbar = tqdm(total=len(self.graphs), desc="Merge Detection")
            for key, graph in self.graphs.items():
                if graph.number_of_nodes() > 0:
                    kdtree = KDTree(graph.voxels)
                    self.count_merges(key, kdtree)
                pbar.update(1)

        # Adjust metrics (if applicable)
        if self.preexisting_merges:
            for key in self.graphs:
                self.adjust_metrics(key)

        # Detect merges by finding ground truth graphs with common node labels
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
            Unique identifier of graph to detect merges.
        kdtree : scipy.spatial.KDTree
            A KD-tree built from voxels in graph corresponding to "key".

        Returns
        -------
        None

        """
        for label in self.get_node_labels(key):
            nodes = self.graphs[key].nodes_with_label(label)
            if len(nodes) > 50:
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
            Unique identifier of graph to detect merges.
        label : int
            Label contained in "labels" attribute in the graph corresponding
            to "key".
        kdtree : scipy.spatial.KDTree
            A KD-tree built from voxels in graph corresponding to "key".

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
                if self.save_merges:
                    fragment_graph.to_zipped_swc(self.zip_writer[key])
                break

    def adjust_metrics(self, key):
        """
        Adjusts the metrics of the graph associated with the given key by
        removing nodes corresponding to known merges and their corresponding
        subgraphs. Updates the total number of edges and run lengths in the
        graph.

        Parameters
        ----------
        key : str
            Unique identifier of the graph to adjust attributes that are are
            used to compute various metrics.

        Returns
        -------
        None

        """
        visited = set()
        for label in self.preexisting_merges:
            label = self.label_handler.mapping[label]
            if label in self.graphs[key].get_labels():
                if label not in visited and label != 0:
                    # Get component with label
                    nodes = self.graphs[key].nodes_with_label(label)
                    root = util.sample_once(list(nodes))

                    # Adjust metrics
                    rl = self.graphs[key].run_length_from(root)
                    self.graphs[key].run_length -= np.sum(rl)
                    self.graphs[key].graph["n_edges"] -= len(nodes) - 1

                    # Update graph
                    self.graphs[key].remove_nodes_from(nodes)
                    visited.add(label)

    def find_label_intersections(self):
        """
        Detects merges between ground truth graphs, namely distinct graphs that
        contain nodes with the same label.

        Parameters
        ----------
        None

        Returns
        -------
        Set[tuple]
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
        return self.label_handler.inverse_mapping[label]

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

    # -- Helpers --
    def find_graph_from_label(self, label):
        for key in self.fragment_graphs:
            if label == util.get_segment_id(key):
                return self.fragment_graphs[key]
        return None

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
