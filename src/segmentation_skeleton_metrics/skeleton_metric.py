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
        (3) Split Rate
        (4) Merge Rate
        (5) Omit Edge Ratio
        (6) Split Edge Ratio
        (7) Merged Edge Ratio
        (8) Edge accuracy
        (9) Expected Run Length (ERL)
        (10) Normalized ERL
    """

    def __init__(
        self,
        gt_pointer,
        label_mask,
        output_dir,
        anisotropy=(1.0, 1.0, 1.0),
        connections_path=None,
        fragments_pointer=None,
        save_merges=False,
        save_fragments=False,
        use_anisotropy=False,
        valid_labels=None,
        verbose=True
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
            Predicted segmentation.
         output_dir : str
            Path to directory wehere results are written.
        anisotropy : Tuple[float], optional
            Image to physical coordinate scaling factors applied to SWC files
            stored at "fragments_pointer". Default is (1.0, 1.0, 1.0).
        connections_path : str, optional
            Path to a txt file containing pairs of segment IDs that represents
            fragments that were merged. Default is None.
        fragments_pointer : Any, optional
            Pointer to SWC files corresponding to "label_mask", see
            "swc_util.Reader" for documentation. Notes: (1) "anisotropy" is
            applied to these SWC files and (2) these SWC files are required
            for counting merges. Default is None.
        save_merges: bool, optional
            Indication of whether to save fragments with a merge mistake.
            Default is None.
        save_fragments : bool, optional
            Indication of whether to save fragments that project onto each
            ground truth skeleton. Default is False.
        valid_labels : Set[int], optional
            Segment IDs that can be assigned to nodes. This argument accounts
            for segments that were been removed due to some type of filtering.
            Default is None.
        use_anisotropy : bool, optional
            Indication of whether coordinates in fragment SWC files should be
            converted from physical to image coordinates using the given
            anisotropy. Default is False.
        verbose : bool, optional
            Indication of whether to printout updates. Default is True.
        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.connections_path = connections_path
        self.output_dir = output_dir
        self.save_merges = save_merges
        self.save_fragments = save_fragments
        self.use_anisotropy = use_anisotropy
        self.verbose = verbose

        # Label handler
        self.label_handler = gutil.LabelHandler(
            connections_path=connections_path, valid_labels=valid_labels
        )

        # Load data
        self.load_groundtruth(gt_pointer, label_mask)
        self.load_fragments(fragments_pointer)

        # Initialize metrics
        util.mkdir(output_dir)
        self.init_writers()
        self.merge_sites = list()

        row_names = list(self.graphs.keys())
        row_names.sort()
        col_names = [
            "SWC Name",
            "SWC Run Length",
            "# Splits",
            "# Merges",
            "Split Rate",
            "Merge Rate",
            "% Split Edges",
            "% Omit Edges",
            "% Merged Edges",
            "Edge Accuracy",
            "ERL",
            "Normalized ERL",
        ]
        self.metrics = pd.DataFrame(np.nan, index=row_names, columns=col_names)
        self.metrics["# Merges"] = 0
        self.metrics["# Splits"] = 0
        self.metrics["SWC Name"] = self.metrics.index

    # --- Load Data ---
    def load_groundtruth(self, swc_pointer, label_mask):
        """
        Loads ground truth graphs and initializes the "graphs" attribute.

        Parameters
        ----------
        swc_pointer : Any
            Pointer to ground truth SWC files.
        label_mask : ImageReader
            Predicted segmentation mask.
        """
        if self.verbose:
            print("\n(1) Load Ground Truth")

        # Build graphs
        graph_loader = gutil.GraphLoader(
            anisotropy=self.anisotropy,
            is_groundtruth=True,
            label_handler=self.label_handler,
            label_mask=label_mask,
            use_anisotropy=False,
        )
        self.graphs = graph_loader.run(swc_pointer)

        # Save initial graphs (if applicable)
        if self.save_merges:
            self.gt_graphs = deepcopy(self.graphs)

    def load_fragments(self, swc_pointer):
        """
        Loads fragments generated from the segmentation and initializes the
        "fragment_graphs" attribute.

        Parameters
        ----------
        swc_pointer : Any
            Pointer to predicted SWC files if provided.
        """
        if self.verbose:
            print("\n(2) Load Fragments")

        if swc_pointer:
            graph_loader = gutil.GraphLoader(
                anisotropy=self.anisotropy,
                is_groundtruth=False,
                selected_ids=self.get_all_node_labels(),
                use_anisotropy=self.use_anisotropy,
            )
            self.fragment_graphs = graph_loader.run(swc_pointer)
            self.set_fragment_ids()
        else:
            self.fragment_graphs = None

    def set_fragment_ids(self):
        """
        Sets the "fragment_ids" attribute by extracting unique segment IDs
        from the "fragment_graphs" keys.
        """
        self.fragment_ids = set()
        for key in self.fragment_graphs:
            self.fragment_ids.add(util.get_segment_id(key))

    def get_all_node_labels(self):
        """
        Gets the set of unique node labels across all graphs in "self.graphs".

        Returns
        -------
        Set[int]
            Unique node labels across all graphs.
        """
        all_node_labels = set()
        inverse_bool = self.label_handler.use_mapping()
        for key in self.graphs:
            node_labels = self.get_node_labels(key, inverse_bool=inverse_bool)
            all_node_labels = all_node_labels.union(node_labels)
        return all_node_labels

    def get_node_labels(self, key, inverse_bool=False):
        """
        Gets the set of unique node labels from the graph corresponding to the
        given key.

        Parameters
        ----------
        key : str
            Unique identifier of graph from which to retrieve the node labels.
        inverse_bool : bool, optional
            Indication of whether to return the labels (from "labels_mask") or
            a remapping of these labels in the case when "connections_path" is
            provided. Default is False.

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
            if os.path.exists(zip_path):
                os.remove(zip_path)
            self.merge_writer = ZipFile(zip_path, "a")

    # -- Main Routine --
    def run(self):
        """
        Computes skeleton-based metrics.
        """
        if self.verbose:
            print("\n(3) Evaluation")

        # Compute metrics
        self.detect_splits()
        self.detect_merges()
        self.compute_edge_accuracy()
        self.compute_erl()

        # Save results
        prefix = "corrected-" if self.connections_path else ""
        path = f"{self.output_dir}/{prefix}results.csv"
        if self.fragment_graphs is None:
            self.metrics = self.metrics.drop("# Merges", axis=1)
            self.metrics = self.metrics.drop("Merge Rate", axis=1)
        self.metrics.to_csv(path, index=False)

        # Report results
        path = os.path.join(self.output_dir, f"{prefix}results-overview.txt")
        util.update_txt(path, "Average Results...")
        for column_name in self.metrics.columns:
            if column_name != "SWC Run Length" and column_name != "SWC Name":
                avg = self.compute_weighted_avg(column_name)
                util.update_txt(path, f"  {column_name}: {avg:.4f}")

        n_splits = self.metrics["# Splits"].sum()
        util.update_txt(path, "\nTotal Results...")
        util.update_txt(path, "  # Splits: " + str(n_splits))

        if self.fragment_graphs is not None:
            n_merges = self.metrics["# Merges"].sum()
            util.update_txt(path, "  # Merges: " + str(n_merges))

    # -- Split Detection --
    def detect_splits(self):
        """
        Perform split detection across all graphs, update graph structures,
        and compute several skeleton metrics.
        """
        if self.verbose:
            pbar = tqdm(total=len(self.graphs), desc="Split Detection")

        with ProcessPoolExecutor(max_workers=4) as executor:
            # Assign processes
            pending = dict()
            for key, graph in self.graphs.items():
                process = executor.submit(split_detection.run, graph)
                pending[process] = key

            # Store results
            for process in as_completed(pending.keys()):
                # Update graph
                key = pending.pop(process)
                graph, n_split_edges = process.result()
                self.graphs[key] = graph

                # Compute metrics
                self.compute_split_metrics(key, n_split_edges)
                if self.verbose:
                    pbar.update(1)

    def compute_split_metrics(self, key, n_split_edges):
        # Edge counts
        n_edges_before = self.graphs[key].graph["n_initial_edges"]
        n_edges_after = self.graphs[key].number_of_edges()
        n_edge_missing = n_edges_before - n_edges_after

        # Split statistics
        n_splits = gutil.count_splits(self.graphs[key])
        p_omit = 100 * (n_edge_missing + n_split_edges) / n_edges_before
        p_split = 100 * n_split_edges / n_edges_before

        # Run lengths
        rl = np.sum(self.graphs[key].run_lengths())
        gt_rl = self.graphs[key].run_length
        split_rate = rl / n_splits if n_splits > 0 else np.nan

        # Record metrics
        self.metrics.at[key, "# Splits"] = n_splits
        self.metrics.at[key, "Split Rate"] = split_rate
        self.metrics.at[key, "% Split Edges"] = round(p_split, 2)
        self.metrics.at[key, "% Omit Edges"] = round(p_omit, 2)
        self.metrics.at[key, "SWC Run Length"] = round(gt_rl, 2)

    # -- Merge Detection --
    def detect_merges(self):
        """
        Detects merges in the predicted segmentation, then deletes node and
        edges in "self.graphs" that correspond to a merge.
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

        # Detect merges by finding ground truth graphs with common node labels
        for (key_1, key_2), label in self.detect_merge_conflicts():
            self.process_merge(key_1, label, -1)
            self.process_merge(key_2, label, -1)

        for key, label, xyz in self.merged_labels:
            self.process_merge(key, label, xyz, update_merged_labels=False)

        self.quantify_merges()

    def count_merges(self, key, kdtree):
        """
        Counts the number of label merges for a given graph key based on
        whether the fragment graph corresponding to a label has a node that is
        more that 40ums away from the nearest point in
        "kdtree".

        Parameters
        ----------
        key : str
            Unique identifier of graph to detect merges.
        kdtree : scipy.spatial.KDTree
            A KD-tree built from voxels in graph corresponding to "key".
        """
        # Iterate over fragments that intersect with GT graphs
        for label in self.get_node_labels(key):
            nodes = self.graphs[key].nodes_with_label(label)
            if len(nodes) > 60:
                for label in self.label_handler.get_class(label):
                    if label in self.fragment_ids:
                        self.check_fragment_for_merges(key, label, kdtree)

    def check_fragment_for_merges(self, key, label, kdtree):
        """
        Checks whether the fragment corresponding to "label" has a merge
        mistake. A fragment has a merge mistake if it has a leaf node more
        than 40μm away from the ground-truth graph corresponding to "key".

        Parameters
        ----------
        key : str
            Unique identifier of graph to detect merges.
        label : int
            Label contained in "labels" attribute in the graph corresponding
            to "key".
        kdtree : scipy.spatial.KDTree
            KD-tree built from voxels in graph corresponding to "key".
        """
        for fragment_graph in self.find_graph_from_label(label):
            if fragment_graph.run_length < 10 ** 6:
                # Search for leaf far from ground truth
                visited = set()
                for leaf in gutil.get_leafs(fragment_graph):
                    voxel = fragment_graph.voxels[leaf]
                    gt_voxel = util.kdtree_query(kdtree, voxel)
                    if self.physical_dist(gt_voxel, voxel) > 40:
                        self.find_merge_site(
                            key, kdtree, fragment_graph, leaf, visited
                        )

                # Save fragment (if applicable)
                if self.save_fragments:
                    gutil.write_graph(
                        fragment_graph, self.fragment_writer[key]
                    )
            else:
                segment_id = util.get_segment_id(fragment_graph.filename)
                run_length = fragment_graph.run_length
                self.merged_labels.add((key, segment_id, -1))
                print(
                    f"Skipping {segment_id} - run_length={run_length}"
                )

    def find_merge_site(self, key, kdtree, fragment_graph, source, visited):
        for _, node in nx.dfs_edges(fragment_graph, source=source):
            if node not in visited:
                # Find closest point in ground truth
                visited.add(node)
                voxel = fragment_graph.voxels[node]
                gt_voxel = util.kdtree_query(kdtree, voxel)
                if self.physical_dist(gt_voxel, voxel) < 3.5:
                    # Local search
                    node = self.branch_search(fragment_graph, kdtree, node)
                    voxel = fragment_graph.voxels[node]

                    # Log merge mistake
                    if self.is_valid_merge(fragment_graph, kdtree, node):
                        filename = fragment_graph.filename
                        segment_id = util.get_segment_id(filename)
                        xyz = img_util.to_physical(voxel, self.anisotropy)
                        self.merged_labels.add((key, segment_id, xyz))
                        self.merge_sites.append(
                            {
                                "Segment_ID": segment_id,
                                "GroundTruth_ID": key,
                                "Voxel": tuple([int(t) for t in voxel]),
                                "World": tuple(
                                    [float(round(t, 2)) for t in xyz]
                                ),
                            }
                        )

                        # Save merged fragment (if applicable)
                        if self.save_merges:
                            gutil.write_graph(
                                fragment_graph, self.merge_writer
                            )
                            gutil.write_graph(
                                self.gt_graphs[key], self.merge_writer
                            )
                        return None

    def is_valid_merge(self, graph, kdtree, root):
        n_hits = 0
        queue = list([(root, 0)])
        visited = set({root})
        while queue:
            # Visit node
            i, d_i = queue.pop()
            voxel_i = graph.voxels[i]
            gt_voxel = util.kdtree_query(kdtree, voxel_i)
            if self.physical_dist(gt_voxel, voxel_i) < 5:
                n_hits += 1

            # Check whether to break
            if n_hits > 16:
                break

            # Update queue
            for j in graph.neighbors(i):
                voxel_j = graph.voxels[j]
                d_j = d_i + self.physical_dist(voxel_i, voxel_j)
                if j not in visited and d_j < 30:
                    queue.append((j, d_j))
                    visited.add(j)
        return True if n_hits > 16 else False

    def process_merge_sites(self):
        if self.merge_sites:
            # Remove duplicates
            idxs = set()
            pts = [s["World"] for s in self.merge_sites]
            for idx_1, idx_2 in KDTree(pts).query_pairs(40):
                idxs.add(idx_1)
            self.merge_sites = pd.DataFrame(self.merge_sites).drop(idxs)

            # Save merge sites
            if self.save_merges:
                row_names = list()
                for i in range(len(self.merge_sites)):
                    filename = f"merge-{i + 1}.swc"
                    xyz = self.merge_sites.iloc[i]["World"]
                    swc_util.to_zipped_point(self.merge_writer, filename, xyz)
                    row_names.append(filename)
                self.merge_sites.index = row_names
                self.merge_writer.close()

            # Update counter
            for key in self.graphs.keys():
                idx_mask = self.merge_sites["GroundTruth_ID"] == key
                n_merges = int(idx_mask.sum())
                rl = np.sum(self.graphs[key].run_lengths())
                merge_rate = rl / n_merges if n_merges > 0 else np.nan

                self.metrics.loc[key, "# Merges"] = n_merges
                self.metrics.loc[key, "Merge Rate"] = merge_rate

            # Save results
            path = os.path.join(self.output_dir, "merge_sites.csv")
            self.merge_sites.to_csv(path, index=True)

    def detect_merge_conflicts(self):
        """
        Detects pairs of distinct graphs that contain nodes that share the
        same label.

        Returns
        -------
        Set[Tuple[Frozenset[str], int]]
            Set of merge conflicts, where each entry is a tuple containing:
            - Frozenset of two graph IDs (the graphs that share a label)
            - label that is present in both graphs
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
        key : str
            Key associated with the graph to be searched.
        label : int
            Label in prediction that is assocatied with a merge.
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
        """
        for key in self.graphs:
            n_initial_edges = self.graphs[key].graph["n_initial_edges"]
            p = self.n_merged_edges[key] / max(n_initial_edges, 1)
            self.metrics.loc[key, "% Merged Edges"] = round(100 * p, 2)

    # -- Compute Metrics --
    def compute_edge_accuracy(self):
        """
        Computes the edge accuracy of each self.graph.
        """
        for key in self.graphs:
            p_omit = self.metrics.loc[key, "% Omit Edges"]
            p_merged = self.metrics.loc[key, "% Merged Edges"]
            edge_accuracy = round(100 - p_omit - p_merged, 2)
            self.metrics.loc[key, "Edge Accuracy"] = edge_accuracy

    def compute_erl(self):
        """
        Computes the expected run length (ERL) of each graph.
        """
        total_run_length = 0
        for key in self.graphs:
            run_length = self.graphs[key].run_length
            run_lengths = self.graphs[key].run_lengths()
            total_run_length += run_length
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            erl = np.sum(wgt * run_lengths)
            n_erl = round(erl / max(run_length, 1), 4)
            self.metrics.loc[key, "ERL"] = round(erl, 2)
            self.metrics.loc[key, "Normalized ERL"] = n_erl

    def compute_weighted_avg(self, column_name):
        # Extract values
        values = self.metrics[column_name]
        weights = self.metrics["SWC Run Length"]

        # Ignore NaNs
        mask = values.notna() & weights.notna()
        values = values[mask]
        weights = weights[mask]

        # Compute weighted mean
        if weights.sum() == 0:
            return float("nan")
        else:
            return (values * weights).sum() / weights.sum()

    # -- Helpers --
    def branch_search(self, graph, kdtree, root, radius=100):
        """
        Searches for a branching node within distance "radius" from the given
        root node.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.
        kdtree : scipy.spatial.KDTree
            KDTree containing voxel coordinates from a ground truth tracing.
        root : int
            Root of search.
        radius : float, optional
            Distance to search from root. Default is 100.

        Returns
        -------
        int
            Root node or closest branching node within distance "radius".
        """
        queue = list([(root, 0)])
        visited = set({root})
        while queue:
            # Visit node
            i, d_i = queue.pop()
            voxel_i = graph.voxels[i]
            if graph.degree[i] > 2:
                gt_voxel = util.kdtree_query(kdtree, voxel_i)
                if self.physical_dist(gt_voxel, voxel_i) < 16:
                    return i

            # Update queue
            for j in graph.neighbors(i):
                voxel_j = graph.voxels[j]
                d_j = d_i + self.physical_dist(voxel_i, voxel_j)
                if j not in visited and d_j < radius:
                    queue.append((j, d_j))
                    visited.add(j)
        return root

    def find_graph_from_label(self, label):
        graphs = list()
        for key in self.fragment_graphs:
            if label == util.get_segment_id(key):
                graphs.append(self.fragment_graphs[key])
        return graphs

    def physical_dist(self, voxel1, voxel2):
        """
        Computes the physical distance between the given voxel coordinates.

        Parameters
        ----------
        voxel1 : Tuple[int]
            Voxel coordinate.
        voxel2 : Tuple[int]
            Voxel coordinate.

        Returns
        -------
        float
            Physical distance between the given voxel coordinates.
        """
        xyz_1 = img_util.to_physical(voxel1, self.anisotropy)
        xyz_2 = img_util.to_physical(voxel2, self.anisotropy)
        return distance.euclidean(xyz_1, xyz_2)

    def to_local_voxels(self, key, i, offset):
        voxel = np.array(self.graphs[key].voxels[i])
        offset = np.array(offset)
        return tuple(voxel - offset)
