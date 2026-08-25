"""
Created on Mon Oct 20 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of class that computes skeleton-based metrics by comparing a
predicted neuron segmentation to a set of ground truth graphs.

"""

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict, deque
from scipy.spatial import KDTree
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd

from segmentation_skeleton_metrics.utils import graph_util as gutil


class SkeletonMetric(ABC):
    """
    Abstract base class for skeleton-based evaluation metrics.
    """

    def __init__(self, verbose=True):
        """
        Instantiates a SkeletonMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Instance attributes
        self.verbose = verbose

    @abstractmethod
    def __call__(self, gt_graphs):
        """
        Abstract method to be implemented by the subclasses.
        """
        pass

    def get_iterator(self, iterator):
        """
        Gets the iterator wrapped in a progress bar if "verbose" is True,
        otherwise returns the iterator unchanged.

        Parameters
        ----------
        iterator : iterable
            Object to be iterated over.

        Returns
        -------
        iterable
            Object to be iterated over.
        """
        return tqdm(iterator, desc=self.name) if self.verbose else iterator

    def reformat(self, results):
        """
        Converts a dictionary of results into a pandas DataFrame.

        Parameters
        ----------
        results : Dict[str, float]
            Dictionary where keys will become the DataFrame index and values
            are used as the single column data.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        results = pd.DataFrame.from_dict(
            results, orient="index", columns=[self.name]
        )
        return results


# --- Subclasses ---
class SplitEdgePercentMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes the percentage of split edges.
    """

    def __init__(self, verbose=True):
        """
        Instantiates a SplitEdgePercentMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "% Split Edges"

    def __call__(self, gt_graphs):
        """
        Computes the percentage of split edges in the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        results = dict()
        for name, graph in self.get_iterator(gt_graphs.items()):
            num_split_edges = self.count_split_edges(graph)
            results[name] = 100 * num_split_edges / graph.number_of_edges()
        return self.reformat(results)

    @staticmethod
    def count_split_edges(graph):
        """
        Counts the number of split edges in the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph to be evaluated.

        Returns
        -------
        num_split_edges : int
            Number fo split edges in the given graph.
        """
        num_split_edges = 0
        for i, j in nx.dfs_edges(graph):
            is_different = graph.node_label[i] != graph.node_label[j]
            is_nonzero = graph.node_label[i] and graph.node_label[j]
            if is_different and is_nonzero:
                num_split_edges += 1
        return num_split_edges


class OmitEdgePercentMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes the percentage of omit edges.
    """

    def __init__(self, verbose=True):
        """
        Instantiates an OmitEdgePercentMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "% Omit Edges"

    def __call__(self, gt_graphs):
        """
        Computes the percentage of omit edges in the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        results = dict()
        for name, graph in self.get_iterator(gt_graphs.items()):
            num_omit_edges = self.count_omit_edges(graph)
            omit_edge_percent = 100 * num_omit_edges / graph.number_of_edges()
            results[name] = omit_edge_percent
        return self.reformat(results)

    @staticmethod
    def count_omit_edges(graph):
        """
        Counts the number of omit edges in the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph to be evaluated.

        Returns
        -------
        num_omit_edges : int
            Number fo omit edges in the given graph.
        """
        num_omit_edges = 0
        for i, j in nx.dfs_edges(graph):
            if graph.node_label[i] == "0" or graph.node_label[j] == "0":
                num_omit_edges += 1
        return num_omit_edges


class MergedEdgePercentMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes the percentage of edges that
    are associated with a merge mistake.
    """

    def __init__(self, verbose=True):
        """
        Instantiates a MergedEdgePercentMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "% Merged Edges"

    def __call__(self, gt_graphs):
        """
        Computes the percentage of merged edges in the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graph to be evaluated.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        # Find graphs with common labels
        self.detect_label_intersections(gt_graphs)

        # Compile results
        results = dict()
        for name, graph in gt_graphs.items():
            # Count number of edges associated with a merge
            num_merged_edges = 0
            for label in graph.labels_with_merge:
                num_merged_edges += len(graph.nodes_with_label(label)) - 1

            # Compute result
            percent = 100 * num_merged_edges / graph.number_of_edges()
            results[name] = percent
        return self.reformat(results)

    def detect_label_intersections(self, gt_graphs):
        """
        Detects pairs of distinct graphs that contain nodes that share the
        same label.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be searched for intersecting labels.
        """
        # Build inverted index: label -> graphs containing it
        label_to_graphs = defaultdict(list)
        for name, graph in self.get_iterator(gt_graphs.items()):
            for label in graph.node_labels():
                label_to_graphs[label].append(graph)

        # Flag label as merge in every graph that has sufficient coverage
        for label, graphs in label_to_graphs.items():
            if len(graphs) < 2:
                continue
            large = [g for g in graphs if len(g.nodes_with_label(label)) > 50]
            if len(large) >= 2:
                for g in large:
                    g.labels_with_merge.add(label)


class SplitCountMetric(SkeletonMetric):
    """
    A skeleton metric subclass that counts the number of splits.
    """

    def __init__(self, verbose=True):
        """
        Instantiates a SplitCountMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "# Splits"

    def __call__(self, gt_graphs):
        """
        Counts the number of split mistakes in each of the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.

        Results
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        results = dict()
        for name, graph in self.get_iterator(gt_graphs.items()):
            num_splits = max(len(graph.node_labels()) - 1, 0)
            results[name] = int(num_splits)
        return self.reformat(results)


class MergeCountMetric(SkeletonMetric):
    """
    A skeleton metric subclass that counts the number merges.
    """

    dist_away_threshold = 50

    def __init__(self, verbose=True):
        """
        Instantiates a MergeCountMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.fragments_with_merge = set()
        self.merge_sites = list()
        self.name = "# Merges"

    # --- Core Routines ---
    def __call__(self, gt_graphs, fragment_graphs):
        """
        Counts the number of split merges in each of the given ground truth
        graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.
        fragment_graphs : Dict[str, FragmentGraph]
            Graphs corresponding to the predicted segmentation.

        Results
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        # Build label -> fragment lookup once
        label_to_fragments = defaultdict(list)
        for fragment_graph in fragment_graphs.values():
            label_to_fragments[fragment_graph.label].append(fragment_graph)

        # Main
        self.merge_sites = list()
        for gt_graph in self.get_iterator(gt_graphs.values()):
            # Build ground truth kd-tree
            gt_graph.set_kdtree()

            # Search intersecting fragments
            for label in gt_graph.node_labels():
                for fragment_graph in label_to_fragments.get(label, []):
                    self.search_for_merges(gt_graph, fragment_graph)

        # Postprocess merge sites
        self.remove_repeat_merge_sites()

        # Compile results
        results = dict()
        for name in gt_graphs:
            if len(self.merge_sites) > 0:
                num_merges = (self.merge_sites["GroundTruth_ID"] == name).sum()
            else:
                num_merges = 0
            results[name] = num_merges
        return self.reformat(results)

    def search_for_merges(self, gt_graph, fragment_graph):
        """
        Searches for potential merge errors in a fragment graph by comparing
        it to a ground truth graph.

        Parameters
        ----------
        gt_graph : LabeledGraph
            Graph to be evaluated.
        fragment_graph : FragmentGraph
            Graph corresponding to a segment in the predicted segmentation.
        """
        # Build a KD-tree from only the gt_graph nodes that carry this
        # fragment's class label. Using the full gt_graph KD-tree would cause
        # false positives when split correction maps several fragments to the
        # same class: leaves of an innocent fragment (one that does not
        # actually overlap this gt_graph) would still appear far from the full
        # skeleton, and adjacency between ground-truth neurons would let the
        # BFS find a close node and falsely record a merge.
        labeled_nodes = gt_graph.nodes_with_label(fragment_graph.label)
        if len(labeled_nodes) == 0:
            return
        labeled_xyz = gt_graph.node_voxel[labeled_nodes][:, [2, 1, 0]] * gt_graph.anisotropy
        labeled_kdtree = KDTree(labeled_xyz)

        visited = set()
        for leaf in fragment_graph.leafs():
            # Check whether to visit
            if leaf in visited or visited.add(leaf):
                continue

            # Find closest labeled node in ground truth
            xyz = fragment_graph.node_xyz(leaf)
            dist, _ = labeled_kdtree.query(xyz)

            # Check if distance to labeled ground truth flags a merge mistake
            if dist > MergeCountMetric.dist_away_threshold:
                self.find_merge_site(
                    gt_graph, fragment_graph, leaf, visited,
                    labeled_nodes, labeled_kdtree
                )

    def find_merge_site(
        self, gt_graph, fragment_graph, source, visited, labeled_nodes, labeled_kdtree
    ):
        """
        Traverses fragment graph from a source node to locate and verify
        potential merge sites relative to the ground truth graph.

        Parameters
        ----------
        gt_graph : LabeledGraph
            Graphs to be evaluated.
        fragment_graphs : FragmentGraph
            Graph corresponding to a segment in the predicted segmentation.
        source : int
            Starting node ID in the fragment graph from which to begin
            traversal.
        visited : Set[int]
            Node IDs from "fragment_graphs" that have already been visited,
            used to avoid redundant exploration.
        labeled_nodes : numpy.ndarray
            Node IDs in gt_graph whose label matches fragment_graph.label.
        labeled_kdtree : scipy.spatial.KDTree
            KD-tree built from the physical coordinates of labeled_nodes.
        """
        queue = deque([source])
        visited.add(source)
        while queue:
            # Visit node
            i = queue.pop()
            xyz_i = fragment_graph.node_xyz(i)
            dist_i, labeled_idx = labeled_kdtree.query(xyz_i)
            if dist_i < 6:
                gt_node = labeled_nodes[labeled_idx]
                self.verify_site(
                    gt_graph, fragment_graph, gt_node, i, labeled_kdtree
                )
                break

            # Update queue
            for j in fragment_graph.neighbors(i):
                if j not in visited:
                    queue.append(j)
                    visited.add(j)

    def verify_site(
        self, gt_graph, fragment_graph, gt_node, fragment_node, labeled_kdtree
    ):
        """
        Verifies whether a given site in a fragment graph corresponds to a
        merge mistake relative to the ground truth graph. If so, the site is
        saved in an internal data structure.

        Parameters
        ----------
        gt_graph : LabeledGraph
            Graph to be evaluated.
        fragment_graph : FragmentGraph
            Graph corresponding to a segment in the predicted segmentation.
        gt_node : int
            Node ID in the ground truth graph corresponding to the site.
        fragment_node : int
            Node ID in the fragment graph corresponding to the candidate site.
        labeled_kdtree : scipy.spatial.KDTree
            KD-tree built from gt_graph nodes whose label matches
            fragment_graph.label.
        """
        # Check if pass through site without merge mistake
        if self.is_nonmerge_pass_thru(fragment_graph, labeled_kdtree):
            return

        # Move site to nearby branching point if possible
        fragment_node = gutil.search_branching_node(
            fragment_graph, gt_graph.kdtree, fragment_node
        )

        # Record site as merge mistake
        voxel = fragment_graph.node_voxel[fragment_node]
        xyz = fragment_graph.node_xyz(fragment_node)

        gt_graph.labels_with_merge.add(fragment_graph.label)
        self.fragments_with_merge.add(fragment_graph.name)
        self.merge_sites.append(
            {
                "Fragment_Name": fragment_graph.name,
                "Segment_ID": fragment_graph.segment_id,
                "GroundTruth_ID": gt_graph.name,
                "Label": fragment_graph.label,
                "Voxel": tuple(map(int, voxel)),
                "World": tuple([float(round(t, 2)) for t in xyz]),
                "Added Cable Length (μm)": 0.0,
            }
        )

    def is_nonmerge_pass_thru(self, fragment_graph, labeled_kdtree):
        """
        Determines whether the fragment has only a small spatial overlap with
        the labeled region of the ground truth graph, indicating a likely
        non-merge pass-through.

        Parameters
        ----------
        fragment_graph : FragmentGraph
            Graph corresponding to a segment in the predicted segmentation.
        labeled_kdtree : scipy.spatial.KDTree
            KD-tree built from gt_graph nodes whose label matches
            fragment_graph.label.

        Returns
        -------
        bool
            True if the fragment has fewer than 50 nodes within
            dist_away_threshold of the labeled gt_graph region, indicating a
            likely pass-through rather than a genuine merge mistake.
        """
        xyz_arr = fragment_graph.node_voxel[:, [2, 1, 0]] * fragment_graph.anisotropy
        dists, _ = labeled_kdtree.query(xyz_arr)
        return np.sum(dists < MergeCountMetric.dist_away_threshold) < 50

    # --- Helpers ---
    def add_merge_site_names(self):
        """
        Assigns unique name to detected merge sites.
        """
        row_names = list()
        for i, _ in enumerate(self.merge_sites.index, 1):
            row_names.append(f"merge-{i}.swc")
        self.merge_sites.index = row_names
        self.merge_sites.index.name = "Merge_ID"

    def remove_repeat_merge_sites(self):
        """
        Removes spatially redundant merge sites within a fixed distance
        threshold.
        """
        if len(self.merge_sites) > 0:
            # Build kdtree from merge sites
            kdtree = KDTree([s["World"] for s in self.merge_sites])

            # Search for repeat sites
            rm_idxs = set()
            for i, site in enumerate(self.merge_sites):
                if i not in rm_idxs:
                    idxs = kdtree.query_ball_point(site["World"], 30)
                    idxs.remove(i)
                    rm_idxs |= set(idxs)

            # Remove repeat sites
            self.merge_sites = pd.DataFrame(self.merge_sites).drop(rm_idxs)
            self.add_merge_site_names()
        else:
            self.merge_sites = pd.DataFrame()


class ERLMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes the expected run length (ERL).
    """

    def __init__(self, verbose):
        """
        Instantiates an ERL object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "ERL"

    def __call__(self, gt_graphs):
        """
        Comptues the expected run length (ERL) of the given graphs.

        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        results = dict()
        for name, graph in self.get_iterator(gt_graphs.items()):
            results[name] = round(self.compute_graph_erl(graph), 2)
        return self.reformat(results)

    @staticmethod
    def compute_graph_erl(graph):
        """
        Computes the ERL of the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph to be evaluated.

        Returns
        -------
        float
            ERL of the given graph.
        """
        wgts = list()
        run_lengths = list()
        for label in graph.node_labels():
            # Compute run length for label
            nodes = graph.nodes_with_label(label)
            run_length = graph.run_length_from(nodes[0])
            graph.labeled_run_length += run_length

            # Update
            wgts.append(run_length)
            run_lengths.append(
                0 if label in graph.labels_with_merge else run_length
            )
        return np.average(run_lengths, weights=wgts) if len(wgts) > 0 else 0


# --- Derived Skeleton Metrics ---
class SplitRateMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes split rate as µm / num_splits.
    """

    def __init__(self, verbose=True):
        """
        Instantiates a SplitRateMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Split Rate"

    def __call__(self, gt_graphs, results):
        """
        Computes split rates for the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.
        results : pandas.DataFrame
            DataFrame containing the skeleton metric results computed so far.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        new_results = dict()
        for name, graph in self.get_iterator(gt_graphs.items()):
            if results["# Splits"][name] > 0:
                rl = gutil.compute_segmented_run_length(graph, results, name)
                new_results[name] = round(rl / results["# Splits"][name], 2)
            else:
                new_results[name] = np.nan
        return self.reformat(new_results)


class MergeRateMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes merge rate as µm / num_merges.
    """

    def __init__(self, verbose=True):
        """
        Instantiates a MergeRateMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Merge Rate"

    def __call__(self, gt_graphs, results):
        """
        Computes merge rates for the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.
        results : pandas.DataFrame
            Data frame containing the skeleton metric results computed so far.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        new_results = dict()
        for name, graph in self.get_iterator(gt_graphs.items()):
            if results["# Merges"][name] > 0:
                rl = gutil.compute_segmented_run_length(graph, results, name)
                new_results[name] = round(rl / results["# Merges"][name], 2)
            else:
                new_results[name] = np.nan
        return self.reformat(new_results)


class EdgeAccuracyMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes edge accuracy.
    """

    def __init__(self, verbose=True):
        """
        Instantiates an EdgeAccuracyMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Edge Accuracy"

    def __call__(self, gt_graphs, results):
        """
        Computes the edge accuracy of the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.
        results : pandas.DataFrame
            DataFrame containing the skeleton metric results computed so far.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are dictionary keys and values are
            stored under a column called "self.name".
        """
        new_results = dict()
        for idx in self.get_iterator(results.index):
            edge_accuracy = 100 - (
                results["% Split Edges"].loc[idx]
                + results["% Omit Edges"].loc[idx]
                + results["% Merged Edges"].loc[idx]
            )
            new_results[idx] = round(edge_accuracy, 2)
        return self.reformat(new_results)


class NormalizedERLMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes normalized expected run
    length (ERL).
    """

    def __init__(self, verbose=True):
        """
        Instantiates a NormalizedERLMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Normalized ERL"

    def __call__(self, gt_graphs, results):
        """
        Computes the normalized ERL of the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.
        results : pandas.DataFrame
            DataFrame containing the skeleton metric results computed so far.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        new_results = dict()
        for name, graph in self.get_iterator(gt_graphs.items()):
            normalized_erl = results["ERL"][name] / graph.run_length
            new_results[name] = round(normalized_erl, 4)
        return self.reformat(new_results)


class AddedCableLengthMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes added cable length.
    """

    def __init__(self, verbose=True):
        """
        Instantiates an AddedCableLengthMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Added Cable Length (μm)"

    def __call__(self, gt_graphs, fragment_graphs, merge_sites):
        """
        Computes the normalized ERL of the given graphs.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.
        fragment_graphs : Dict[str, FragmentGraph]
            Graphs corresponding to the predicted segmentation.
        merge_sites : pandas.DataFrame
            Data frame containing detected merge sites.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame where the indices are the dictionary keys and values are
            stored under a column called "self.name".
        """
        # Check if merge sites is empty
        if len(merge_sites) == 0:
            return None

        # Compute metric
        pair_to_length = dict()
        for i in self.get_iterator(merge_sites.index):
            # Extract site info
            gt_id = merge_sites["GroundTruth_ID"][i]
            label = merge_sites["Label"][i]
            name = merge_sites["Fragment_Name"][i]
            pair_id = (label, gt_id)

            # Check whether to visit
            if pair_id in pair_to_length:
                merge_sites.loc[i, self.name] = pair_to_length[pair_id]
            else:
                # Get graphs
                gt_graph = gt_graphs[gt_id]
                if label in fragment_graphs:
                    fragment_graph = deepcopy(fragment_graphs[label])
                elif name in fragment_graphs:
                    fragment_graph = deepcopy(fragment_graphs[name])

                # Compute metric
                pair_to_length[pair_id] = self.compute_added_length(
                    gt_graph, fragment_graph
                )
                merge_sites.loc[i, self.name] = pair_to_length[pair_id]

    def compute_added_length(self, gt_graph, fragment_graph):
        """
        Computes the total cable length of fragment components that are not
        sufficiently close to the ground-truth graph.

        Parameters
        ----------
        gt_graph : LabeledGraph
            Graph containing merge mistake.
        fragment_graph : FragmentGraph
            Fragment that is merged to the given ground truth graph.

        Returns
        -------
        cable_length : float
            Total cable length of fragment components that remain after pruning
            nodes near the ground-truth graph.
        """
        # Remove nodes close to ground truth
        xyz_arr = fragment_graph.node_voxel * fragment_graph.anisotropy
        dists, _ = gt_graph.kdtree.query(xyz_arr)
        max_dist = MergeCountMetric.dist_away_threshold
        fragment_graph.remove_nodes_from(np.where(dists < max_dist)[0])

        # Compute cable length
        cable_length = 0
        for nodes in map(list, nx.connected_components(fragment_graph)):
            cable_length += fragment_graph.run_length_from(nodes[0])
        return round(float(cable_length), 2)
