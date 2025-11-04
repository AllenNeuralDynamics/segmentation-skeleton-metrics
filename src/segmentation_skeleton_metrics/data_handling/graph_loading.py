"""
Created on Thu Oct 16 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module for loading graph structures, labeling nodes, and handling label
management.

"""

from collections import deque
from concurrent.futures import (
    as_completed, ProcessPoolExecutor, ThreadPoolExecutor
)
from tqdm import tqdm

import networkx as nx
import numpy as np

from segmentation_skeleton_metrics.data_handling import swc_loading
from segmentation_skeleton_metrics.data_handling.graph_classes import (
    FragmentGraph, LabeledGraph
)
from segmentation_skeleton_metrics.utils import util


class DataLoader:
    """
    A class that loads ground truth and fragment graphs and provides tools for
    labeling ground truth graphs.
    """

    def __init__(
        self,
        label_handler,
        anisotropy=(1.0, 1.0, 1.0),
        use_anisotropy=False,
        verbose=True
    ):
        """
        Instantiates a DataLoader object.

        Parameters
        ----------
        label_handler : LabelHander
            Handles mapping between raw segmentation labels and consolidated
            class IDs.
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        use_anisotropy : bool, optional
            Indication of whether to apply the anisotropy to the coordinates
            from the fragment SWC files. Default is False.
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.label_handler = label_handler
        self.use_anisotropy = use_anisotropy
        self.verbose = verbose

    # --- Core Routines ---
    def load_groundtruth(self, swc_pointer, label_mask):
        """
        Loads ground truth graphs.

        Parameters
        ----------
        swc_pointer : str
            Pointer to ground truth SWC files.
        label_mask : ImageReader
            Predicted segmentation.

        Returns
        -------
        Dict[str, SkeletonGraph]
            Ground truth graphs.
        """
        if self.verbose:
            print("\n(1) Load Ground Truth")

        graph_loader = GraphLoader(
            anisotropy=self.anisotropy,
            is_groundtruth=True,
            label_handler=self.label_handler,
            label_mask=label_mask,
            use_anisotropy=False,
            verbose=self.verbose
        )
        return graph_loader.run(swc_pointer)

    def load_fragments(self, swc_pointer, gt_graphs):
        """
        Loads fragment graphs (predicted skeletons).

        Parameters
        ----------
        swc_pointer : str
            Path or pointer to predicted SWC files.
        gt_graphs : Dict[str, SkeletonGraph]
            Ground truth graphs to extract node labels from.

        Returns
        -------
        Dict[str, SkeletonGraph] or None
            Fragment graphs or None.
        """
        if self.verbose:
            print("\n(2) Load Fragments")

        # Check if SWC pointer is provided
        if not swc_pointer:
            return None

        # Load fragments
        selected_ids = self.get_all_node_labels(gt_graphs)
        graph_loader = GraphLoader(
            anisotropy=self.anisotropy,
            is_groundtruth=False,
            label_handler=self.label_handler,
            selected_ids=selected_ids,
            use_anisotropy=self.use_anisotropy,
            verbose=self.verbose
        )
        return graph_loader.run(swc_pointer)

    # --- Helpers ---
    def get_all_node_labels(self, graphs):
        """
        Gets the set of unique node labels across all given graphs.

        Parameters
        ----------
        graphs : Dict[str, SkeletonGraph]
            Graph to be searched.

        Returns
        -------
        node_labels : Set[int]
            Unique node labels across all graphs.
        """
        node_labels = set()
        for graph in graphs.values():
            node_labels |= self.label_handler.get_node_labels(graph)
        return node_labels


class GraphLoader:
    """
    A class that builds graphs from SWC files.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        is_groundtruth=False,
        label_handler=None,
        label_mask=None,
        selected_ids=None,
        use_anisotropy=True,
        verbose=True
    ):
        """
        Instantiates a GraphLoader object.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        is_groundtruth : bool, optional
            Indication of whether this graph corresponds to a ground truth
            tracing. Default is False.
        label_mask : ImageReader, optional
            Predicted segmentation mask.
        selected_ids : Set[int], optional
            Only SWC files with a name contained in this set are read. Default
            is None.
        use_anisotropy : bool, optional
            Indication of whether coordinates in SWC files should be converted
            from physical to image coordinates using the given anisotropy.
            Default is True.
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.is_groundtruth = is_groundtruth
        self.label_handler = label_handler
        self.label_mask = label_mask
        self.verbose = verbose

        # Reader
        anisotropy = anisotropy if use_anisotropy else (1.0, 1.0, 1.0)
        self.swc_reader = swc_loading.Reader(
            anisotropy, selected_ids=selected_ids
        )

    def run(self, swc_pointer):
        """
        Builds a graphs by reading SWC files to extract content to load into a
        SkeletonGraph object. Nodes are labeled if a label_mask is provided.

        Parameters
        ----------
        swc_pointer : Any
            Object that points to SWC files to be read.

        Returns
        -------
        graphs : Dict[str, SkeletonGraph]
            Dictionary where the keys are unique identifiers (i.e. filenames
            of SWC files) and values are the corresponding SkeletonGraph.
        """
        graphs = self._build_graphs_from_swcs(swc_pointer)
        if self.label_mask:
            for name in graphs:
                self._label_graph(graphs[name])
        return graphs

    # --- Build Graphs ---
    def _build_graphs_from_swcs(self, swc_pointer):
        """
        Builds graphs by reading SWC files to extract content which is then
        loaded into a custom SkeletonGraph object.

        Parameters
        ----------
        swc_pointer : Any
            Object that points to SWC files to be read.

        Returns
        -------
        graphs : Dict[str, SkeletonGraph]
            Dictionary where the keys are unique identifiers (i.e. SWC
            filenames) and values are the corresponding SkeletonGraphs.
        """
        # Initializations
        swc_dicts = self.swc_reader.read(swc_pointer)
        if self.verbose:
            pbar = tqdm(total=len(swc_dicts), desc="Build Graphs")

        # Main
        graphs = dict()
        if len(swc_dicts) > 10 ** 4:
            while len(swc_dicts) > 0:
                swc_dict = swc_dicts.pop()
                graphs.update(self.to_graph(swc_dict))
                if self.verbose:
                    pbar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=1) as executor:
                # Assign processes
                processes = list()
                while len(swc_dicts) > 0:
                    swc_dict = swc_dicts.pop()
                    processes.append(executor.submit(self.to_graph, swc_dict))

                # Store results
                for process in as_completed(processes):
                    graphs.update(process.result())
                    if self.verbose:
                        pbar.update(1)
        return graphs

    def to_graph(self, swc_dict):
        """
        Builds a graph from a dictionary that contains the contents of an SWC
        file.

        Parameters
        ----------
        swc_dict : dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.

        Returns
        -------
        Dict[str, SkeletonGraph]
            Graph built from an SWC file.
        """
        # Initialize graph
        graph = self._init_graph(swc_dict)

        # Build graph structure
        id_lookup = dict()
        for i, id_i in enumerate(swc_dict["id"]):
            id_lookup[id_i] = i
            if swc_dict["pid"][i] != -1:
                parent = id_lookup[swc_dict["pid"][i]]
                graph.add_edge(i, parent)
                graph.run_length += graph.dist(i, parent)
        graph.prune_branches()
        return {graph.name: graph}

    def _init_graph(self, swc_dict):
        """
        Initializes and returns a graph object from a parsed SWC dictionary.

        Parameters
        ----------
        swc_dict : dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.

        Returns
        -------
        graph : SkeletonGraph
            An initialized LabeledGraph or FragmentGraph instance with voxel
            data, filename, and node count set.
        """
        # Instantiate graph
        if self.is_groundtruth:
            graph = LabeledGraph(
                anisotropy=self.anisotropy, name=swc_dict["swc_name"]
            )
        else:
            segment_id = util.get_segment_id(swc_dict["swc_name"])
            label = self.label_handler.get(segment_id)
            graph = FragmentGraph(
                anisotropy=self.anisotropy,
                name=swc_dict["swc_name"],
                label=label,
                segment_id=segment_id
            )

        # Set class attributes
        graph.init_voxels(swc_dict["voxel"])
        graph.set_filename(swc_dict["swc_name"] + ".swc")
        graph.set_nodes(len(swc_dict["id"]))
        return graph

    # --- Label Graphs ---
    def _label_graph(self, graph):
        """
        Assigns labels to graph nodes by indexing a segmentation mask using
        each node’s voxel coordinates.

        Parameters
        ----------
        graph : LabeledGraph
            Graph to be labeled.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            batch = set()
            threads = list()
            visited = set()
            for i, j in nx.dfs_edges(graph):
                # Check whether to start new batch
                if len(batch) == 0:
                    root = i
                    batch.add(i)
                    visited.add(i)

                # Check whether to submit batch
                is_node_far = graph.dist(root, j) > 128
                is_batch_full = len(batch) >= 128
                if is_node_far or is_batch_full:
                    threads.append(
                        executor.submit(self.get_patch_labels, graph, batch)
                    )
                    batch = set()

                # Visit j
                if j not in visited:
                    batch.add(j)
                    visited.add(j)
                    if len(batch) == 1:
                        root = j

            # Submit last batch
            threads.append(
                executor.submit(self.get_patch_labels, graph, batch)
            )

            # Store results
            graph.init_node_labels()
            for thread in as_completed(threads):
                node_to_label = thread.result()
                for i, label in node_to_label.items():
                    graph.node_labels[i] = label

        GraphLoader.fix_label_misalignments(graph)

    def get_patch_labels(self, graph, nodes):
        """
        Gets the segment labels for a given set of nodes within a specified
        patch of the label mask.

        Parameters
        ----------
        graph : str
            Unique identifier of graph to be labeled.
        nodes : List[int]
            Node IDs for which the labels are to be retrieved.

        Returns
        -------
        node_to_label : Dict[int, int]
            Dictionary that maps node IDs to their respective labels.
        """
        bbox = graph.get_bbox(nodes)
        label_patch = self.label_mask.read_with_bbox(bbox)
        node_to_label = dict()
        for i in nodes:
            voxel = self.to_local_voxels(graph, i, bbox["min"])
            label = self.label_handler.get(label_patch[voxel])
            node_to_label[i] = label
        return node_to_label

    def to_local_voxels(self, graph, i, offset):
        """
        Converts a global voxel coordinate to a local voxel coordinate.

        Parameters
        ----------
        graph : SkeletonGraph
            Graph object containing node voxel coordinates.
        i : int
            Node ID of voxel coordinate to be converted.
        offset : ArrayLike
            Offset to subtract from the global voxel coordinate to get the
            local coordinate.

        Returns
        -------
        Tuple[int]
            Local voxel coordinate after subtracting the offset.
        """
        voxel = np.array(graph.voxels[i])
        offset = np.array(offset)
        return tuple(voxel - offset)

    @staticmethod
    def fix_label_misalignments(graph):
        """
        Adjusts misalignments between the labeled graph and segmentation.

        Parameters
        ----------
        graph : LabeledGraph
            Graph to be searched.
        """
        visited_edges = set()
        for i, j in deque(nx.dfs_edges(graph)):
            # Check whether to visit edge
            if frozenset({i, j}) in visited_edges:
                continue

            # Visit edge
            if int(graph.node_labels[j]) == 0:
                GraphLoader.check_misalignment(graph, visited_edges, i, j)
            visited_edges.add(frozenset({i, j}))

    @staticmethod
    def check_misalignment(graph, visited_edges, nb, root):
        """
        Determines whether zero-valued label corresponds to a misalignment
        between the graph and segmentation mask.

        Parameters
        ----------
        graph : LabeledGraph
            Graph that represents a ground truth neuron.
        visited_edges : List[Frozenset[int]]
            List of edges in "graph" that have been visited.
        nb : int
            Neighbor of "root".
        root : int
            Node where possible misalignment starts (i.e. zero-valued label).
        """
        # Search graph
        label_collisions = set()
        queue = deque([root])
        visited = set()
        while len(queue) > 0:
            # Visit node
            j = queue.popleft()
            label_j = int(graph.node_labels[j])
            if label_j != 0:
                label_collisions.add(label_j)
            visited.add(j)

            # Update queue
            if label_j == 0:
                for k in graph.neighbors(j):
                    if k not in visited:
                        if frozenset({j, k}) not in visited_edges or k == nb:
                            queue.append(k)
                            visited_edges.add(frozenset({j, k}))

        # Upd zero nodes
        if len(label_collisions) == 1:
            label = label_collisions.pop()
            graph.update_node_labels(visited, label)


class LabelHandler:
    """
    Handles mapping between raw segmentation labels and consolidated class IDs.

    The class is designed to manage cases where multiple segment IDs are merged
    into a single equivalence class. It supports:
      - Building mappings from a file of pairwise segment connections.
      - Mapping individual labels to class IDs.
      - Retrieving all labels belonging to a given class.
      - Enforcing constraints on which labels are considered valid.

    Attributes
    ----------
    mapping : Dict[int, int]
        Maps a raw label (segment ID) to its class ID.
    inverse_mapping : Dict[int, Set[int]]
        Maps a class ID back to the set of raw labels it contains.
    processed_labels : Set[int]
        Labels that have been processed during initialization.
    valid_labels : Set[int]
        Labels that are allowed to be assigned (after filtering).
    """

    def __init__(self, connections_path=None, valid_labels=set()):
        """
        Instantiates a LabelHandler object and optionally builds label
        mappings.

        Parameters
        ----------
        connections_path : str, optional
            Path to file containing pairs of segment IDs that were merged.
            Default is None.
        valid_labels : Set[int], optional
            Subset of labels that are considered to be valid. This argument
            accounts for segments removed due to filtering. Default is an
            empty set.
        """
        self.mapping = dict()  # Maps label to equivalent class id
        self.inverse_mapping = dict()  # Maps class id to list of labels
        self.processed_labels = set()
        self.valid_labels = valid_labels
        if connections_path:
            self.init_mappings(connections_path)

    # --- Constructor Helpers ---
    def init_mappings(self, connections_path):
        """
        Initializes dictionaries that map between segment IDs and equivalent
        class IDS.

        Parameters
        ----------
        connections_path : str
            Path to file containing pairs of segment IDs that were merged.
        """
        self.mapping = {0: 0}
        self.inverse_mapping = {0: [0]}
        labels_graph = self.build_labels_graph(connections_path)
        for i, labels in enumerate(nx.connected_components(labels_graph)):
            class_id = i + 1
            self.inverse_mapping[class_id] = set()
            for label in labels:
                self.mapping[label] = class_id
                self.inverse_mapping[class_id].add(label)

    def build_labels_graph(self, connections_path):
        """
        Builds a graph of labels from valid labels and merge connections.
        Nodes correspond to "self.valid_labels", and edges are added between
        labels that were merged according to the file.

        Parameters
        ----------
        connections_path : str
            Path to a text file containing merge connections. Each line should
            specify a pair of segment IDs separated by a comma.

        Returns
        -------
        labels_graph : networkx.Graph
            Graph with nodes that represent labels and edges are based on the
            connections read from the "connections_path".
        """
        # Initializations
        assert self.valid_labels is not None, "Must provide valid labels!"
        labels_graph = nx.Graph()
        labels_graph.add_nodes_from(self.valid_labels)

        # Main
        for line in util.read_txt(connections_path):
            ids = line.split(",")
            id_1 = util.get_segment_id(ids[0])
            id_2 = util.get_segment_id(ids[1])
            labels_graph.add_edge(id_1, id_2)
        return labels_graph

    # --- Core Routines ---
    def get(self, label):
        """
        Maps a raw label to its class ID.

        Parameters
        ----------
        label : int
            Raw label (segment ID) to be mapped.

        Returns
        -------
        int
            Class ID corresponding to the label.
        """
        if self.use_mapping():
            return self.mapping.get(label, 0)
        elif self.valid_labels:
            return 0 if label not in self.valid_labels else label
        return label

    def get_class(self, label):
        """
        Gets all raw labels associated with a class ID.

        Parameters
        ----------
        label : int
            Class ID or raw label.

        Returns
        -------
        List[int]
            Labels corresponding to the class.
        """
        return self.inverse_mapping[label] if self.use_mapping() else [label]

    def use_mapping(self):
        """
        Checks whether mappings have been initialized.

        Returns
        -------
        bool
            True if mappings are active, False otherwise.
        """
        return True if len(self.mapping) > 0 else False

    # --- Helpers ---
    def get_node_labels(self, graph):
        """
        Gets the set of unique node labels from the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph from which to retrieve the node labels.

        Returns
        -------
        labels : Set[int]
            Labels corresponding to nodes in the graph identified by "key".
        """
        labels = graph.get_node_labels()
        if self.use_mapping():
            labels = set().union(*(self.inverse_mapping[l] for l in labels))
        return labels
