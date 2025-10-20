"""
Created on Mon Oct 20 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of class that computes skeleton-based metrics by comparing a
predicted neuron segmentation to a set of ground truth graphs.

"""

from abc import ABC, abstractmethod
from scipy.spatial import KDTree
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd

from segmentation_skeleton_metrics.utils import util


class SkeletonMetric(ABC):
    """
    Abstract base class for skeleton-based evaluation metrics.
    """

    def __init__(self, verbose=True):
        # Instance attributes
        self.verbose = verbose

    @abstractmethod
    def compute(self, gt_graphs):
        """
        Abstract method to be implemented by the subclasses.
        """
        pass

    def get_pbar(self, total):
        return tqdm(total=total, desc=self.name) if self.verbose else None

    def reformat(self, results):
        results = pd.DataFrame.from_dict(
            results, orient="index", columns=[self.name]
        )
        return results


# --- Subclasses ---
class SplitEdgePercentMetric(SkeletonMetric):
    """
    A skeleton metric subclass that compute the percentage of split edges.
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

    def compute(self, gt_graphs):
        results = dict()
        pbar = self.get_pbar(len(gt_graphs))
        for name, graph in gt_graphs.items():
            # Compute result
            num_split_edges = self.count_split_edges(graph)
            percent = 100 * num_split_edges / graph.number_of_edges()
            results[name] = percent

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(results)

    @staticmethod
    def count_split_edges(graph):
        """
        Counts the number of split edges in the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph to be searched.

        Returns
        -------
        num_split_edges : int
            Number fo split edges in the given graph.
        """
        num_split_edges = 0
        for i, j in nx.dfs_edges(graph):
            is_different = graph.node_labels[i] != graph.node_labels[j]
            is_nonzero = graph.node_labels[i] and graph.node_labels[j]
            if is_different and is_nonzero:
                num_split_edges += 1
        return num_split_edges


class OmitEdgePercentMetric(SkeletonMetric):
    """
    A skeleton metric subclass that compute the percentage of omit edges.
    """

    def __init__(self, verbose=True):
        """
        Instantiates a OmitEdgePercentMetric object.

        Parameters
        ----------
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "% Omit Edges"

    def compute(self, gt_graphs):
        results = dict()
        pbar = self.get_pbar(len(gt_graphs))
        for name, graph in gt_graphs.items():
            # Compute result
            num_omit_edges = self.count_omit_edges(graph)
            omit_edge_percent = 100 * num_omit_edges / graph.number_of_edges()
            results[name] = omit_edge_percent

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(results)

    @staticmethod
    def count_omit_edges(graph):
        """
        Counts the number of omit edges in the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph to be searched.

        Returns
        -------
        num_omit_edges : int
            Number fo omit edges in the given graph.
        """
        num_omit_edges = 0
        for i, j in nx.dfs_edges(graph):
            if graph.node_labels[i] == 0 and graph.node_labels[j] == 0:
                num_omit_edges += 1
        return num_omit_edges


class MergedEdgePercentMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes the percentage of edges that
    are associated with a merge mistake.
    """

    def __init__(self, verbose=True):
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "% Merged Edges"

    def compute(self, gt_graphs):
        # Find graphs with common labels
        self.detect_label_intersections(gt_graphs)

        # Compile results
        results = dict()
        for name, graph in gt_graphs.items():
            # Count number of edges associated with a merge
            num_merged_edges = 0
            for label in graph.labels_with_merge:
                num_merged_edges += len(graph.get_nodes_with_label(label)) - 1

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
        gt_graphs : LabeledGraph
            Ground truth graphs to be searched for intersecting labels.
        """
        visited = set()
        pbar = self.get_pbar(len(gt_graphs))
        for name1, graph1 in gt_graphs.items():
            # Search other graphs for label intersections
            for name2, graph2 in gt_graphs.items():
                names = frozenset((name1, name2))
                if name1 != name2 and names not in visited:
                    visited.add(names)
                    labels1 = set(graph1.get_node_labels())
                    labels2 = set(graph2.get_node_labels())
                    for label in labels1.intersection(labels2):
                        # Check if intersection is meaningful
                        num_nodes1 = len(graph1.get_nodes_with_label(label))
                        num_nodes2 = len(graph2.get_nodes_with_label(label))
                        if num_nodes1 > 50 and num_nodes2 > 50:
                            graph1.labels_with_merge.add(label)
                            graph2.labels_with_merge.add(label)

            # Update progress bar
            if self.verbose:
                pbar.update(1)


class SplitCountMetric(SkeletonMetric):
    """
    A skeleton metric subclass that counts the number of splits.
    """

    def __init__(self, verbose=True):
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "# Splits"

    def compute(self, gt_graphs):
        results = dict()
        pbar = self.get_pbar(len(gt_graphs))
        for name, graph in gt_graphs.items():
            # Compute result
            num_splits = len(graph.get_node_labels()) - 1
            results[name] = int(num_splits)

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(results)


class MergeCountMetric(SkeletonMetric):
    """
    A skeleton metric subclass that counts the number merges.
    """

    def __init__(self, verbose=True):
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.fragments_with_merge = set()
        self.merge_sites = list()
        self.name = "# Merges"

    # --- Core Routines ---
    def compute(self, gt_graphs, fragment_graphs):
        # Main
        pbar = self.get_pbar(len(gt_graphs))
        for gt_graph in gt_graphs.values():
            # Build ground truth kd-tree
            gt_graph.init_kdtree(use_voxels=False)

            # Search intersecting fragments
            labels = gt_graph.get_node_labels()
            for fragment_graph in fragment_graphs.values():
                if fragment_graph.label in labels:
                    self.search_for_merges(gt_graph, fragment_graph)

            # Update progress bar
            if self.verbose:
                pbar.update(1)

        # Postprocess merge sites
        self.remove_repeat_merge_sites()

        # Compile results
        results = dict()
        for name in gt_graphs:
            num_merges = (self.merge_sites["GroundTruth_ID"] == name).sum()
            results[name] = num_merges
        return self.reformat(results)

    def search_for_merges(self, gt_graph, fragment_graph):
        """
        Searches for potential merge errors in a fragment graph by comparing
        it to a ground truth graph.

        Parameters
        ----------
        gt_graph : SkeletonGraph
            Ground truth graph.
        fragment_graph : SkeletonGraph
            Fragment graph to be checked for merges.
        """
        visited = set()
        for leaf in util.get_leafs(fragment_graph):
            # Check whether to visit
            if leaf in visited or visited.add(leaf):
                continue

            # Find closet node in ground truth
            xyz = fragment_graph.get_xyz(leaf)
            dist, _ = gt_graph.kdtree.query(xyz)

            # Check if distance to ground truth flags a merge mistake
            if dist > 40:
                self.find_merge_site(gt_graph, fragment_graph, leaf, visited)

    def find_merge_site(self, gt_graph, fragment_graph, source, visited):
        """
        Traverses fragment graph from a source node to locate and verify
        potential merge sites relative to the ground truth graph.

        Parameters
        ----------
        gt_graph : SkeletonGraph
            Ground truth graph.
        fragment_graph : SkeletonGraph
            Fragment graph to be checked for merges.
        source : int
            Starting node ID in the fragment graph from which to begin
            traversal.
        visited : Set[int]
            Node IDs that have already been visited, used to avoid redundant
            exploration.
        """
        # Traverse until close to ground truth
        for _, node in nx.dfs_edges(fragment_graph, source=source):
            # Check whether to visit
            if node in visited or visited.add(node):
                continue

            # Check if close to ground truth
            xyz = fragment_graph.get_xyz(node)
            dist, gt_node = gt_graph.kdtree.query(xyz)
            if dist < 6:
                self.verify_site(gt_graph, fragment_graph, gt_node, node)
                break

    def verify_site(self, gt_graph, fragment_graph, gt_node, fragment_node):
        # Check if pass through site without merge mistake
        if self.is_nonmerge_pass_thru(gt_graph, fragment_graph, gt_node):
            return

        # Move site to nearby branching point if possible
        fragment_node = util.search_branching_node(
            fragment_graph, gt_graph.kdtree, fragment_node
        )

        # Record site as merge mistake
        voxel = fragment_graph.voxels[fragment_node]
        xyz = fragment_graph.get_xyz(fragment_node)

        self.fragments_with_merge.add(fragment_graph.name)
        self.merge_sites.append(
            {
                "Segment_ID": fragment_graph.segment_id,
                "GroundTruth_ID": gt_graph.name,
                "Voxel": tuple(map(int, voxel)),
                "World": tuple([float(round(t, 2)) for t in xyz]),
            }
        )

    def is_nonmerge_pass_thru(self, gt_graph, fragment_graph, gt_node):
        nodes = gt_graph.get_nodes_with_label(fragment_graph.label)
        subgraph = gt_graph.subgraph(nodes)
        for nodes_cc in nx.connected_components(subgraph):
            if gt_node in nodes_cc:
                return len(nodes_cc) < 50
        return True

    # --- Helpers ---
    def add_merge_site_names(self):
        row_names = list()
        for i, _ in enumerate(self.merge_sites.index, 1):
            row_names.append(f"merge-{i + 1}.swc")
        self.merge_sites.index = row_names

    def remove_repeat_merge_sites(self):
        if len(self.merge_sites) > 0:
            # Build kdtree from merge sites
            kdtree = KDTree([s["World"] for s in self.merge_sites])

            # Search for repeat sites
            rm_idxs = set()
            for i, site in enumerate(self.merge_sites):
                if i not in rm_idxs:
                    idxs = kdtree.query_ball_point(site["World"], 40)
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
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "ERL"

    def compute(self, gt_graphs):
        results = dict()
        pbar = self.get_pbar(len(gt_graphs))
        for name, graph in gt_graphs.items():
            # Compute result
            erl = self.compute_graph_erl(graph)
            results[name] = round(erl, 2)

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(results)

    @staticmethod
    def compute_graph_erl(graph):
        wgts = list()
        run_lengths = list()
        for label in graph.get_node_labels():
            # Compute run length for label
            nodes = graph.get_nodes_with_label(label)
            run_length = graph.run_length_from(nodes[0])
            graph.labeled_run_length += run_length

            # Update
            wgts.append(run_length)
            run_lengths.append(
                0 if label in graph.labels_with_merge else run_length
            )
        return np.average(run_lengths, weights=wgts)


# --- Derived Skeleton Metrics ---
class SplitRateMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes split rate as µm / num_splits.
    """

    def __init__(self, verbose=True):
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Split Rate"

    def compute(self, gt_graphs, results):
        new_results = dict()
        pbar = self.get_pbar(len(results.index))
        for name, graph in gt_graphs.items():
            # Compute result
            if results["# Splits"][name] > 0:
                rate = graph.labeled_run_length / results["# Splits"][name]
            else:
                rate = np.nan
            new_results[name] = round(rate, 2)

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(new_results)


class MergeRateMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes merge rate as µm / num_merges.
    """

    def __init__(self, verbose=True):
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Merge Rate"

    def compute(self, gt_graphs, results):
        new_results = dict()
        pbar = self.get_pbar(len(results.index))
        for name, graph in gt_graphs.items():
            # Compute result
            if results["# Merges"][name] > 0:
                rate = graph.labeled_run_length / results["# Merges"][name]
            else:
                rate = np.nan
            new_results[name] = round(rate, 2)

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(new_results)


class EdgeAccuracyMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes the edge accuracy.
    """

    def __init__(self, verbose=True):
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Edge Accuracy"

    def compute(self, gt_graphs, results):
        new_results = dict()
        pbar = self.get_pbar(len(gt_graphs))
        for idx in results.index:
            # Compute result
            edge_accuracy = 100 - (
                results["% Split Edges"].loc[idx] +
                results["% Omit Edges"].loc[idx] +
                results["% Merged Edges"].loc[idx])
            new_results[idx] = round(edge_accuracy, 2)

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(new_results)


class NormalizedERLMetric(SkeletonMetric):
    """
    A skeleton metric subclass that computes the normalized expected run
    length (ERL).
    """

    def __init__(self, verbose=True):
        # Call parent class
        super().__init__(verbose=verbose)

        # Instance attributes
        self.name = "Normalized ERL"

    def compute(self, gt_graphs, results):
        new_results = dict()
        pbar = self.get_pbar(len(gt_graphs))
        for name, graph in gt_graphs.items():
            # Compute result
            normalized_erl = results["ERL"][name] / graph.run_length
            new_results[name] = round(normalized_erl, 4)

            # Update progress bar
            if self.verbose:
                pbar.update(1)
        return self.reformat(new_results)
