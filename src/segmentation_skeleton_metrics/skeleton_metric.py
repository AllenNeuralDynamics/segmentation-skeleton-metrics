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
from segmentation_skeleton_metrics.swc_utils import save, to_graph, get_xyz_coords
from segmentation_skeleton_metrics.utils import resolve


INTERSECTION_THRESHOLD = 10

detected_sites = [
    [24253.677938, 12223.122055, 3517.545410],
    #[24879.321500, 10933.487512, 2438.581299],
    [24506.729844, 11969.226457, 3229.514648],
    [24086.304172, 13403.438297, 3334.542725],
    [23997.799117, 11885.819344, 3238.000000],
    [24581.471406, 13374.270680, 3865.545898],
    [24331.614570, 12680.798711, 3771.828857],
    [24699.763516, 13129.664453, 3527.499512],
    [24699.763516, 13129.664453, 3527.499512],
    [24733.008609, 13032.409844, 3771.502686],
    [24808.936453, 13065.339375, 3771.418457],
    [23816.039500, 12339.684414, 3611.450684],
    [23975.565109, 12168.482262, 3328.587158],
    [24435.447781, 13312.036203, 3716.725342],
    [24487.278922, 12969.612906, 3768.517334],
    [24134.228766, 13434.386797, 3776.521240],
    [24621.188453, 13542.997273, 3707.002686],
    [24054.875023, 12161.695477, 3282.862305],
]

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
        self.label_mask = pred_labels

        self.anisotropy = anisotropy
        self.ignore_boundary_mistakes = ignore_boundary_mistakes
        self.init_black_holes(black_holes_xyz_id)
        self.black_hole_radius = black_hole_radius

        self.write_to_swc = write_to_swc
        self.output_dir = output_dir

        # Build Graphs
        self.target_graphs = self.init_graphs(target_swc_paths, anisotropy)
        self.labeled_target_graphs = self.init_labeled_target_graphs()
        self.pred_swc_paths = pred_swc_paths

        # Build kdtree
        self.init_xyz_to_id_node()
        self.init_kdtree()
        self.rm_spurious_intersections()

    # -- Initialize and Label Graphs --
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
        "self.label_mask" that coincides with it.

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
        self.id_to_label_nodes = dict() # {target_id: {label: nodes}}
        for cnt, (target_id, graph) in enumerate(self.target_graphs.items()):
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            labeled_target_graph, id_to_label_nodes = self.label_graph(graph)
            labeled_target_graphs[target_id] = labeled_target_graph
            self.id_to_label_nodes[target_id] = id_to_label_nodes

        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")
        return labeled_target_graphs

    def label_graph(self, target_graph):
        """
        Iterates over nodes in "target_graph" and stores the label in the
        predicted segmentation mask (i.e. "self.label_mask") which coincides with
        each node as a node-level attribute called "label".

        Parameters
        ----------
        target_graph : networkx.Graph
            Graph that represents a neuron from the ground truth.

        Returns
        -------
        target_graph : networkx.Graph
            Updated graph with node-level attributes called "label".

        """
        labeled_target_graph = nx.Graph(target_graph)
        id_to_label_nodes = dict()
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = []
            for i in labeled_target_graph.nodes:
                img_coord = gutils.get_coord(labeled_target_graph, i)
                threads.append(executor.submit(self.get_label, img_coord, i))

            # Store results
            for thread in as_completed(threads):
                i, label = thread.result()
                labeled_target_graph.nodes[i].update({"label": label})
                if label in id_to_label_nodes.keys():
                    id_to_label_nodes[label].add(i)
                else:
                    id_to_label_nodes[label] = set([i])
        return labeled_target_graph, id_to_label_nodes

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
        return self.finalize_label(label, return_node)

    def finalize_label(self, label, return_node):
        if return_node:
            return return_node, self.is_valid(label)
        else:
            return self.is_valid(label)

    def __read_label(self, coord):
        """
        Gets label at image coordinates "xyz".

        Parameters
        ----------
        coord : tuple[int]
            Coordinates that indexes into "self.label_mask".

        Returns
        -------
        int
           Label at image coordinates "xyz".

        """
        if type(self.label_mask) == ts.TensorStore:
            return int(self.label_mask[coord].read().result())
        else:
            return self.label_mask[coord]

    def is_valid(self, label):
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

    def init_xyz_to_id_node(self):
        self.xyz_to_id_node = dict()
        for target_id, graph in self.target_graphs.items():
            for i in graph.nodes:
                xyz = tuple(graph.nodes[i]["xyz"])
                if xyz in self.xyz_to_id_node.keys():
                    self.xyz_to_id_node[xyz][target_id] = i
                else:
                    self.xyz_to_id_node[xyz] = {target_id: i}

    def rm_spurious_intersections(self):
        for label in [label for label in self.get_all_labels() if label > 0]:
            # Compute label intersect target_graphs
            hit_target_ids = dict()
            multi_hits = set()
            for xyz in self.get_pred_xyz(label):
                hat_xyz, d = self.get_projection(xyz)
                if d < 5:
                    hits = list(self.xyz_to_id_node[hat_xyz].keys())
                    if len(hits) > 1:
                        multi_hits.add(hat_xyz)
                    else:
                        hat_i = self.xyz_to_id_node[hat_xyz][hits[0]]
                        hit_target_ids = utils.append_dict_value(
                          hit_target_ids, hits[0], hat_i
                        )
            hit_target_ids = utils.resolve(
                multi_hits, hit_target_ids, self.xyz_to_id_node
            )

            # Remove spurious intersections
            for target_id in self.target_graphs.keys():
                if target_id in hit_target_ids.keys():
                    n_hits = len(hit_target_ids[target_id])
                    if n_hits < INTERSECTION_THRESHOLD:
                        self.zero_nodes(target_id, label)
                elif label in self.id_to_label_nodes[target_id]:
                    self.zero_nodes(target_id, label)

    def get_pred_xyz(self, label):
        for path in self.pred_swc_paths:
            swc_id = utils.get_swc_id(path)
            if str(label) == swc_id:
                return get_xyz_coords(path, anisotropy=self.anisotropy)
        return []

    # -- Final Constructor Routines --
    def init_kdtree(self):
        xyz_list = []
        for _, graph in self.target_graphs.items():
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
        self.detect_splits()
        self.quantify_splits()

        # Merge evaluation
        print("Detecting Merges...")
        self.detect_merges()
        self.quantify_merges()

        # Compute metrics
        full_results, avg_results = self.compile_results()
        return full_results, avg_results

    def get_all_labels(self):
        labels = set()
        for target_id in self.target_graphs.keys():
            labels = labels.union(self.get_labels(target_id))
        return labels
            
    def get_labels(self, target_id):
        """
        Gets the predicted label ids that intersect with the target graph
        corresponding to "target_id".

        Parameters
        ----------
        target_id : str

        """
        return set(self.id_to_label_nodes[target_id].keys())

    def zero_nodes(self, target_id, label):
        if label in self.id_to_label_nodes[target_id].keys():
            for i in self.id_to_label_nodes[target_id][label]:
                self.labeled_target_graphs[target_id].nodes[i]["label"] = 0
            self.id_to_label_nodes[target_id][label] = set()

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
        for cnt, (target_id, target_graph) in enumerate(target_graphs):
            # Detection
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            labeled_graph = self.labeled_target_graphs[target_id]
            labeled_graph = split_detection.run(target_graph, labeled_graph)

            # Update predicted graph
            labeled_graph = gutils.delete_nodes(labeled_graph, 0)
            labeled_graph = gutils.delete_nodes(labeled_graph, -1)
            id_to_label_nodes = gutils.store_labels(labeled_graph)
            self.labeled_target_graphs[target_id] = labeled_graph
            self.id_to_label_nodes[target_id] = id_to_label_nodes

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
        for target_id in self.target_graphs.keys():
            n_splits = gutils.count_splits(self.labeled_target_graphs[target_id])
            n_pred_edges = self.labeled_target_graphs[target_id].number_of_edges()
            n_target_edges = self.target_graphs[target_id].number_of_edges()

            self.split_cnts[target_id] = n_splits
            self.omit_cnts[target_id] = n_target_edges - n_pred_edges
            self.omit_percents[target_id] = 1 - n_pred_edges / n_target_edges

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
        for cnt, target_id_1 in enumerate(self.labeled_target_graphs.keys()):
            utils.progress_bar(cnt + 1, len(self.target_graphs))
            for target_id_2 in self.labeled_target_graphs.keys():
                # Check if identical
                if target_id_1 == target_id_2:
                    continue

                # Compare labels contained in graph
                labels_1 = self.get_labels(target_id_1)
                labels_2 = self.get_labels(target_id_2)
                intersection = labels_1.intersection(labels_2)
                for label in intersection:
                    sites, d = self.localize(target_id_1, target_id_2, label)
                    if d < 30 and self.write_to_swc:
                        # Process merge
                        self.save_swc(sites[0], sites[1], "merge")
                        self.process_merge(target_id_1, label)
                        self.process_merge(target_id_2, label)

                        # Remove label to avoid reprocessing
                        del self.id_to_label_nodes[target_id_1][label]
                        del self.id_to_label_nodes[target_id_2][label]

        # Report Runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def localize(self, target_id_1, target_id_2, label):
        # Get merged nodes
        merged_1 = self.id_to_label_nodes[target_id_1][label]
        merged_2 = self.id_to_label_nodes[target_id_2][label]

        # Find closest pair
        min_dist = np.inf
        xyz_pair = [None, None]
        for i in merged_1:
            for j in merged_2:
                xyz_1 = self.target_graphs[target_id_1].nodes[i]["xyz"]
                xyz_2 = self.target_graphs[target_id_2].nodes[j]["xyz"]
                dist = utils.dist(xyz_1, xyz_2)
                if dist < min_dist:
                    min_dist = dist
                    xyz_pair = [xyz_1, xyz_2]
        return xyz_pair, min_dist

    def near_bdd(self, xyz):
        near_bdd_bool = False
        if self.ignore_boundary_mistakes:
            above = [xyz[i] >= self.label_mask.shape[i] - 32 for i in range(3)]
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

    def process_merge(self, target_id, label):
        """
        Once a merge has been detected that corresponds to "target_id", every node
        in "self.labeled_target_graph[target_id]" with that "label" is deleted.

        Parameters
        ----------
        target_id : str
            Key associated with the labeled_target_graph to be searched.
        label : int
            Label in prediction that is assocatied with a merge.

        Returns
        -------
        None

        """
        # Update graph
        graph = self.labeled_target_graphs[target_id].copy()
        graph, merged_cnt = gutils.delete_nodes(graph, label, return_cnt=True)
        self.labeled_target_graphs[target_id] = graph

        # Update cnts
        self.merge_cnts[target_id] += 1
        self.merged_cnts[target_id] += merged_cnt

    def quantify_merges(self):
        """
        Computes the percentage of merged edges for each labeled_target_graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.merged_percents = dict()
        for target_id in self.target_graphs.keys():
            n_edges = self.target_graphs[target_id].number_of_edges()
            percent = self.merged_cnts[target_id] / n_edges
            self.merged_percents[target_id] = percent

    def compile_results(self):
        """
        Compiles a dictionary containing the metrics computed by this module.

        Parameters
        ----------
        None

        Returns
        -------
        full_results : dict
            Dictionary where the keys are target_ids and the values are the result
            of computing each metric for the corresponding graphs.
        avg_result : dict
            Dictionary where the keys are names of metrics computed by this
            module and values are the averaged result over all target_ids.

        """
        # Compute remaining metrics
        self.compute_edge_accuracy()
        self.compute_erl()

        # Summarize results
        target_ids, results = self.generate_full_results()
        avg_results = self.generate_avg_results()

        # Reformat full results
        full_results = dict()
        for i, target_id in enumerate(target_ids):
            full_results[target_id] = dict(
                [(key, results[key][i]) for key in results.keys()]
            )

        return full_results, avg_results

    def generate_full_results(self):
        """
        Generates a report by creating a list of the results for each metric.
        Each item in this list corresponds to a graph in labeled_target_graphs
        and this list is ordered with respect to "target_ids".

        Parameters
        ----------
        None

        Results
        -------
        target_ids : list[str]
            Specifies the ordering of results for each value in "stats".
        stats : dict
            Dictionary where the keys are metrics and values are the result of
            computing that metric for each graph in labeled_target_graphs.

        """
        target_ids = list(self.labeled_target_graphs.keys())
        target_ids.sort()
        stats = {
            "# splits": generate_result(target_ids, self.split_cnts),
            "# merges": generate_result(target_ids, self.merge_cnts),
            "% omit": generate_result(target_ids, self.omit_percents),
            "% merged": generate_result(target_ids, self.merged_percents),
            "edge accuracy": generate_result(target_ids, self.edge_accuracy),
            "erl": generate_result(target_ids, self.erl),
            "normalized erl": generate_result(target_ids, self.normalized_erl),
        }
        return target_ids, stats

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
        for target_id, wgt in self.wgts.items():
            if self.omit_percents[target_id] < 1:
                result.append(stats[target_id])
                wgts.append(wgt)
        return np.average(result, weights=wgts)

    def compute_edge_accuracy(self):
        """
        Computes the edge accuracy of each labeled_target_graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.edge_accuracy = dict()
        for target_id in self.target_graphs.keys():
            omit_percent = self.omit_percents[target_id]
            merged_percent = self.merged_percents[target_id]
            self.edge_accuracy[target_id] = 1 - omit_percent - merged_percent

    def compute_erl(self):
        """
        Computes the expected run length (ERL) of each labeled_target_graph.

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
        for target_id in self.target_graphs.keys():
            labeled_target_graph = self.labeled_target_graphs[target_id]
            target_graph = self.target_graphs[target_id]

            path_length = gutils.compute_path_length(target_graph)
            run_lengths = gutils.compute_run_lengths(labeled_target_graph)
            wgt = run_lengths / max(np.sum(run_lengths), 1)

            self.erl[target_id] = np.sum(wgt * run_lengths)
            self.normalized_erl[target_id] = self.erl[target_id] / path_length

            self.wgts[target_id] = path_length
            total_path_length += path_length

        for target_id in self.target_graphs.keys():
            self.wgts[target_id] = self.wgts[target_id] / total_path_length

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
        
        min_dist = np.inf
        hit_site = None
        for site in detected_sites:
            d = utils.dist(xyz_1, np.array(site))
            if d < min_dist:
                min_dist = d
                hit_site = site

        if min_dist < 10:
            print("")
            print("site:", hit_site)
            print("nn dist:", min_dist)
            print("")

        if mistake_type == "split":
            color = "0.0 1.0 0.0"
            cnt = 1 + np.sum(list(self.split_cnts.values())) // 2
        else:
            color = "0.0 0.0 1.0"
            cnt = 1 + np.sum(list(self.merge_cnts.values())) // 2

        path = f"{self.output_dir}/{mistake_type}-{cnt}.swc"
        save(path, xyz_1, xyz_2, color=color)


# -- utils --
def generate_result(target_ids, stats):
    """
    Reorders items in "stats" with respect to the order defined by
    "target_ids".

    Parameters
    ----------
    target_ids : list[str]
        List of all |target_ids" of graphs in "self.labeled_target_graphs".
    stats : dict
        Dictionary where the keys are "target_ids" and values are the result
        of computing some metrics.

    Returns
    -------
    list
        Reorded items in "stats" with respect to the order defined by
        "target_ids".

    """
    return [stats[target_id] for target_id in target_ids]
