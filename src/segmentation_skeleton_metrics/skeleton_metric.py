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
from copy import deepcopy
from scipy.spatial import distance, KDTree
from tqdm import tqdm
from zipfile import ZipFile

import networkx as nx
import numpy as np
import os
import pandas as pd

from segmentation_skeleton_metrics import split_detection
from segmentation_skeleton_metrics.utils import (
    graph_util as gutil,
    img_util,
    swc_util,
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
        output_dir,
        anisotropy=(1.0, 1.0, 1.0),
        connections_path=None,
        fragments_pointer=None,
        preexisting_merges=None,
        save_merges=False,
        save_fragments=False,
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
         output_dir : str
            Path to directory wehere results are written.
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
        preexisting_merges : List[int], optional
            List of segment IDs that are known to contain a merge mistake. The
            default is None.
        save_merges: bool, optional
            Indication of whether to save fragments with a merge mistake. The
            default is None.
        save_fragments : bool, optional
            Indication of whether to save fragments that project onto each
            ground truth skeleton. The default is False.
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
        self.save_fragments = save_fragments

        # Label handler
        self.label_handler = gutil.LabelHandler(
            connections_path=connections_path, valid_labels=valid_labels
        )

        # Load data
        self.label_mask = label_mask
        self.load_groundtruth(gt_pointer)
        self.load_fragments(fragments_pointer)

        # Initialize metrics
        util.mkdir(output_dir, delete=True)
        self.init_writers()
        self.merge_sites = list()

        row_names = list(self.graphs.keys())
        col_names = [
            "# Splits",
            "# Merges",
            "% Split",
            "% Omit",
            "% Merged",
            "Edge Accuracy",
            "ERL",
            "Normalized ERL",
            "GT Run Length"
        ]
        self.metrics = pd.DataFrame(index=row_names, columns=col_names)

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
        self.gt_graphs = deepcopy(self.graphs)

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

        Parameters
        ----------
        None

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
        Gets the segment labels for a given set of nodes within a specified
        patch of the label mask.

        Parameters
        ----------
        key : str
            Unique identifier of graph to be labeled.
        nodes : List[int]
            Node IDs for which the labels are to be retrieved.

        Returns
        -------
        dict
            A dictionary that maps node IDs to their respective labels.

        """
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
        Gets the set of unique node labels from all graphs in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        Set[int]
            Set of unique node labels from all graphs.

        """
        all_labels = set()
        inverse_bool = self.label_handler.use_mapping()
        for key in self.graphs:
            labels = self.get_node_labels(key, inverse_bool=inverse_bool)
            all_labels = all_labels.union(labels)
        return all_labels

    def get_node_labels(self, key, inverse_bool=False):
        """
        Gets the set of unique node labels from the graph corresponding to the
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

    def init_writers(self):
        """
        Initializes "self.merge_writer" attribute by setting up a directory for
        output files and creating ZIP files for each graph in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Fragments writer
        if self.save_fragments:
            # Initialize directory
            fragments_dir = os.path.join(self.output_dir, "fragments")
            util.mkdir(fragments_dir, delete=True)

            # ZIP writer
            self.fragment_writer = dict()
            for key in self.graphs.keys():
                zip_path = f"{fragments_dir}/{key}.zip"
                self.fragment_writer[key] = ZipFile(zip_path, "w")
                self.graphs[key].to_zipped_swc(self.fragment_writer[key])

        # Merged fragments writer
        if self.save_merges:
            zip_path = os.path.join(self.output_dir, "merged_fragments.zip")
            self.merge_writer = ZipFile(zip_path, "a")

    # -- Main Routine --
    def run(self):
        """
        Computes skeleton-based metrics.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("\n(3) Evaluation")

        # Compute metrics
        self.detect_splits()
        self.detect_merges()
        self.compute_edge_accuracy()
        self.compute_erl()

        # Save results
        prefix = "corrected-" if self.connections_path else ""
        path = f"{self.output_dir}/{prefix}results.csv"
        self.metrics.to_csv(path, index=False)

        # Report results
        path = os.path.join(self.output_dir, f"{prefix}results-overview.txt")
        util.update_txt(path, "Average Results...")
        for column_name in self.metrics.columns:
            if column_name != "GT Run Length":
                avg = self.compute_weighted_avg(column_name)
                util.update_txt(path, f"  {column_name}: {avg:.4f}")

        n_splits = self.metrics["# Splits"].sum()
        n_merges = self.metrics["# Merges"].sum()
        util.update_txt(path, "\nTotal Results...")
        util.update_txt(path, "  # Splits: " + str(n_splits))
        util.update_txt(path, "  # Merges: " + str(n_merges))

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
        with ProcessPoolExecutor(max_workers=4) as executor:
            # Assign processes
            processes = list()
            for key, graph in self.graphs.items():
                processes.append(
                    executor.submit(split_detection.run, key, graph)
                )

            # Store results
            for process in as_completed(processes):
                key, graph, split_percent = process.result()
                n_edges = graph.number_of_edges()
                n_gt_edges = graph.graph["n_edges"]

                self.graphs[key] = graph
                self.metrics.at[key, "% Omit"] = 1 - n_edges / n_gt_edges
                self.metrics.at[key, "# Splits"] = gutil.count_splits(graph)
                self.metrics.loc[key, "% Split"] = split_percent
                self.metrics.loc[key, "GT Run Length"] = graph.run_length
                pbar.update(1)

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
        self.n_merged_edges = {key: 0 for key in self.graphs}
        self.merged_labels = set()

        # Detect merges by comparing fragment graphs to ground truth graphs
        if self.fragment_graphs:
            pbar = tqdm(total=len(self.graphs), desc="Merge Detection")
            for key, graph in self.graphs.items():
                if graph.number_of_nodes() > 0:
                    self.count_merges(key, KDTree(graph.voxels))
                pbar.update(1)
            self.process_merge_sites()

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

        self.quantify_merges()

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
        # Iterate over fragments that intersect with GT skeleton
        for label in tqdm(self.get_node_labels(key), desc="Merge Search"):
            nodes = self.graphs[key].nodes_with_label(label)
            if len(nodes) > 40:
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
        for fragment_graph in self.find_graph_from_label(label):
            if fragment_graph.run_length < 10**6:
                # Search for leaf far from ground truth
                visited = set()
                for leaf in gutil.get_leafs(fragment_graph):
                    voxel = fragment_graph.voxels[leaf]
                    gt_voxel = util.kdtree_query(kdtree, voxel)
                    if self.physical_dist(gt_voxel, voxel) > 50:
                        visited = self.find_merge_site(
                            key, kdtree, fragment_graph, leaf, visited
                        )

                # Save fragment (if applicable)
                if self.save_fragments:
                    for node in fragment_graph.nodes:
                        voxel = fragment_graph.voxels[node]
                        gt_voxel = util.kdtree_query(kdtree, voxel)
                        if self.physical_dist(gt_voxel, voxel) < 3:
                            gutil.write_graph(
                                fragment_graph, self.fragment_writer[key]
                            )
                            break

    def find_merge_site(self, key, kdtree, fragment_graph, source, visited):
        for _, node in nx.dfs_edges(fragment_graph, source=source):
            if node not in visited:
                # Find closest point in ground truth
                visited.add(node)
                voxel = fragment_graph.voxels[node]
                gt_voxel = util.kdtree_query(kdtree, voxel)
                if self.physical_dist(gt_voxel, voxel) < 2:
                    # Log merge mistake
                    segment_id = util.get_segment_id(fragment_graph.filename)
                    xyz = img_util.to_physical(voxel, self.anisotropy)
                    self.merged_labels.add((key, segment_id, xyz))
                    self.merge_sites.append(
                        {
                            "Segment_ID": segment_id,
                            "GroundTruth_ID": key,
                            "Voxel": voxel,
                            "World": xyz,
                        }
                    )

                    # Save merged fragment (if applicable)
                    if self.save_merges:
                        gutil.write_graph(fragment_graph, self.merge_writer)
                        gutil.write_graph(
                             self.gt_graphs[key], self.merge_writer
                         )
                    return visited
        return visited

    def process_merge_sites(self):
        if self.merge_sites:
            # Remove duplicates
            idxs = set()
            pts = [s["World"] for s in self.merge_sites]
            for idx_1, idx_2 in KDTree(pts).query_pairs(30):
                idxs.add(idx_1)
            self.merge_sites = pd.DataFrame(self.merge_sites).drop(idxs)

            # Save merge sites
            for i in range(len(self.merge_sites)):
                filename = f"merge-{i + 1}.swc"
                xyz = self.merge_sites.iloc[i]["World"]
                swc_util.to_zipped_point(self.merge_writer, filename, xyz)

            # Update counter
            for key in self.graphs.keys():
                idx_mask = self.merge_sites["GroundTruth_ID"] == key
                self.metrics.loc[key, "# Merges"] = int(idx_mask.sum())

            # Save results
            path = os.path.join(self.output_dir, "merge_sites.csv")
            self.merge_sites.to_csv(path, index=False)
            self.merge_writer.close()

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
            self.n_merged_edges[key] += subgraph.number_of_edges()

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
        for key in self.graphs:
            p = self.n_merged_edges[key] / self.graphs[key].graph["n_edges"]
            self.metrics.loc[key, "% Merged"] = p

    # -- Compute Metrics --
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
        for key in self.graphs:
            p_omit = self.metrics.loc[key, "% Omit"]
            p_merged = self.metrics.loc[key, "% Merged"]
            self.metrics.loc[key, "Edge Accuracy"] = 1 - p_omit - p_merged

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
        total_run_length = 0
        for key in self.graphs:
            run_length = self.graphs[key].run_length
            run_lengths = self.graphs[key].run_lengths()
            total_run_length += run_length
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            erl = np.sum(wgt * run_lengths)
            self.metrics.loc[key, "ERL"] = erl
            self.metrics.loc[key, "Normalized ERL"] = erl / max(run_length, 1)

    def compute_weighted_avg(self, column_name):
        wgt = self.metrics["GT Run Length"]
        return (self.metrics[column_name] * wgt).sum() / wgt.sum()

    # -- Helpers --
    def find_graph_from_label(self, label):
        graphs = list()
        for key in self.fragment_graphs:
            if label == util.get_segment_id(key):
                graphs.append(self.fragment_graphs[key])
        return graphs

    def physical_dist(self, voxel_1, voxel_2):
        """
        Computes the physical distance between the given voxel coordinates.

        Parameters
        ----------
        voxel_1 : Tuple[int]
            Voxel coordinate.
        voxel_2 : Tuple[int]
            Voxel coordinate.

        Returns
        -------
        float
            Physical distance between the given voxel coordinates.

        """
        xyz_1 = img_util.to_physical(voxel_1, self.anisotropy)
        xyz_2 = img_util.to_physical(voxel_2, self.anisotropy)
        return distance.euclidean(xyz_1, xyz_2)

    def to_local_voxels(self, key, i, offset):
        voxel = np.array(self.graphs[key].voxels[i])
        offset = np.array(offset)
        return tuple(voxel - offset)
