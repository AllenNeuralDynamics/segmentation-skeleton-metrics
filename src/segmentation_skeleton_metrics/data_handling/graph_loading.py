"""
Created on Thu Oct 16 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module for loading graph structures, labeling nodes, and handling label
management.

"""

from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from tqdm import tqdm

import networkx as nx
import numpy as np

from segmentation_skeleton_metrics.data_handling import swc_loading
from segmentation_skeleton_metrics.data_handling.graph_classes import (
    FragmentGraph,
    LabeledGraph,
)
from segmentation_skeleton_metrics.utils import util


class DataLoader:
    """
    A class that loads ground truth and fragment graphs and provides tools for
    labeling ground truth graphs.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        label_handler=None,
        use_anisotropy=False,
        verbose=True,
    ):
        """
        Instantiates a DataLoader object.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        label_handler : LabelHander
            Handles mapping between raw segmentation labels and consolidated
            class IDs.
        use_anisotropy : bool, optional
            Indication of whether coordinates in SWC files should be converted
            from physical to image coordinates using the given anisotropy.
            Default is False.
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.label_handler = label_handler or LabelHandler()
        self.use_anisotropy = use_anisotropy
        self.verbose = verbose

    # --- Core Routines ---
    def load_groundtruth(self, swc_pointer, segmentation):
        """
        Loads ground truth graphs.

        Parameters
        ----------
        swc_pointer : str
            Pointer to ground truth SWC files.
        segmentation : Image
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
            segmentation=segmentation,
            use_anisotropy=self.use_anisotropy,
            verbose=self.verbose,
        )
        return graph_loader(swc_pointer)

    def load_fragments(self, swc_pointer, swc_names=set()):
        """
        Loads fragment graphs (predicted skeletons).

        Parameters
        ----------
        swc_pointer : str
            Path to predicted SWC files.
        swc_names : Set[str], optional
            Only SWC files with names in this set are loaded if provided.
            Otherwise, all SWC files are loaded. Default is None.

        Returns
        -------
        Dict[str, SkeletonGraph]
            Fragment graphs.
        """
        if self.verbose:
            print("\n(2) Load Fragments")

        # Check if SWC pointer is provided
        if not swc_pointer:
            return None

        # Load fragments
        graph_loader = GraphLoader(
            anisotropy=self.anisotropy,
            is_groundtruth=False,
            label_handler=self.label_handler,
            swc_names=swc_names,
            use_anisotropy=self.use_anisotropy,
            verbose=self.verbose,
        )
        return graph_loader(swc_pointer)

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
        node_labels : Set[str]
            Unique node labels across all graphs.
        """
        node_labels = set()
        for graph in graphs.values():
            node_labels |= self.label_handler.node_labels(graph)
        return node_labels


class GraphLoader:
    """
    A class that builds graphs from SWC files.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        fix_label_misalignments=True,
        is_groundtruth=False,
        label_handler=None,
        segmentation=None,
        swc_names=set(),
        use_anisotropy=False,
        verbose=True,
    ):
        """
        Instantiates a GraphLoader object.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        fix_label_misalignments : bool, optional
            Indication of whether to fix misalignments between skeletons and
            segmentation mask. Default is True.
        is_groundtruth : bool, optional
            Indication of whether this graph corresponds to a ground truth
            tracing. Default is False.
        segmentation : Image, optional
            Predicted segmentation mask.
        swc_names : Set[str], optional
            Only SWC files with names in this set are loaded if provided.
            Otherwise, all SWC files are loaded. Default is an empty set.
        use_anisotropy : bool, optional
            Indication of whether coordinates in SWC files should be converted
            from physical to image coordinates using the given anisotropy.
            Default is False.
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        # Instance attributes
        self.anisotropy = np.array(anisotropy)
        self.fix_label_misalignments = fix_label_misalignments
        self.is_groundtruth = is_groundtruth
        self.label_handler = label_handler
        self.segmentation = segmentation
        self.use_anisotropy = use_anisotropy
        self.verbose = verbose

        # Reader
        self.swc_reader = swc_loading.Reader(
            swc_names=swc_names, verbose=verbose
        )

    def __call__(self, swc_pointer):
        """
        Builds a graphs by reading SWC files to extract content to load into a
        SkeletonGraph object. Nodes are labeled if a segmentation is provided.

        Parameters
        ----------
        swc_pointer : str
            Object that points to SWC files to be read.

        Returns
        -------
        graphs : Dict[str, SkeletonGraph]
            Dictionary where the keys are unique identifiers (i.e. filenames
            of SWC files) and values are the corresponding SkeletonGraph.
        """
        graphs = self._build_graphs_from_swcs(swc_pointer)
        if self.segmentation:
            for name in self.iterator(graphs, desc="Label Graphs"):
                self._label_graph(graphs[name])
                if self.fix_label_misalignments:
                    graphs[name].fix_label_misalignments()
        return graphs

    # --- Build Graphs ---
    def _build_graphs_from_swcs(self, swc_pointer):
        """
        Builds graphs by reading SWC files to extract content which is then
        loaded into a custom SkeletonGraph object.

        Parameters
        ----------
        swc_pointer : str
            Path to SWC files to be read.

        Returns
        -------
        graphs : Dict[str, SkeletonGraph]
            Dictionary where the keys are unique identifiers (i.e. SWC
            filenames) and values are the corresponding SkeletonGraphs.
        """
        # Initializations
        swc_dicts = self.swc_reader(swc_pointer)
        pbar = self.manual_progress_bar(len(swc_dicts), desc="Build Graphs")

        # Main
        graphs = dict()
        if len(swc_dicts) > 10**4:
            while len(swc_dicts) > 0:
                swc_dict = swc_dicts.pop()
                graphs.update(self.to_graph(swc_dict))
                if self.verbose:
                    pbar.update(1)
        else:
            with ProcessPoolExecutor() as executor:
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

        # Apply voxel coordinate conversion (if applicable)
        if self.use_anisotropy:
            graph.node_voxel = (graph.node_voxel / self.anisotropy).astype(int)
            graph.node_voxel[:, [0, 2]] = graph.node_voxel[:, [2, 0]]
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
            label = self.get_label(segment_id)
            graph = FragmentGraph(
                anisotropy=self.anisotropy,
                name=swc_dict["swc_name"],
                label=label,
                segment_id=segment_id,
            )

        # Set class attributes
        graph.set_voxels(swc_dict["voxel"])
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
                    graph.node_label[i] = label

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
        patch = self.segmentation.read_with_bbox(bbox)
        node_to_label = dict()
        for i in nodes:
            voxel = tuple(graph.node_voxel[i] - bbox["min"])
            node_to_label[i] = self.get_label(patch[voxel])
        return node_to_label

    # --- Helpers ---
    def get_label(self, segment_id):
        if self.label_handler:
            return self.label_handler.get(segment_id)
        else:
            return segment_id

    def iterator(self, iterator, desc=""):
        """
        Gets an iterator that optionally displays a progress bar.

        Parameters
        ----------
        iterator : iterable
            Object to be iterated over.
        desc : str, optional
            Text to display on progress bar. Default is an empty string.

        Returns
        -------
        tqdm.tqdm
            Iterator that is optionally wrapped in a progress bar.
        """
        return tqdm(iterator, desc=desc) if self.verbose else iterator

    def manual_progress_bar(self, total, desc=""):
        """
        Gets progress bar that needs to be updated manually.

        Parameters
        ----------
        total : int
            Size of progress bar.
        desc : str, optional
            Text to be displayed on progress bar. Default is an empty string.

        Returns
        -------
        tqdm.tqdm
            Iterator that is optionally wrapped in a progress bar.
        """
        return tqdm(total=total, desc=desc) if self.verbose else None


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
    labels : Set[int]
        Labels that are allowed to be assigned (after filtering).
    """

    def __init__(self, labels=set(), label_pairs=set()):
        """
        Instantiates a LabelHandler object and optionally builds label
        mappings.

        Parameters
        ----------
        labels : Set[hashable], optional
            Labels considered to be valid. This argument accounts for segments
            removed due to filtering. Default is an empty set.
        label_pairs : List[hashable], optional
            Pairs of labels merged during split correction. Default is an
            empty set.
        """
        self.labels = labels
        self.mapping = dict()  # Maps label to equivalent class id
        self.inverse_mapping = dict()  # Maps class id to list of labels
        self.init_mappings(label_pairs)

    # --- Constructor Helpers ---
    def init_mappings(self, label_pairs):
        """
        Initializes dictionaries that map between segment IDs and equivalent
        class IDS.

        Parameters
        ----------
        label_pairs : List[hashable]
            Pairs of labels merged during split correction.
        """
        self.mapping = {0: 0}
        self.inverse_mapping = {0: [0]}
        for i, labels in enumerate(self.label_equiv_classes(label_pairs)):
            class_id = i + 1 if label_pairs else labels[0]
            self.inverse_mapping[class_id] = set()
            for label in labels:
                self.mapping[label] = class_id
                self.inverse_mapping[class_id].add(label)

    def label_equiv_classes(self, label_pairs):
        """
        Computes equiavelence classes of labels by building a graph from them
        and computing the connected components.

        Parameters
        ----------
        label_pairs : List[hashable]
            Pairs of labels merged during split correction.

        Returns
        -------
        Iterator[List[hashable]] : networkx.Graph
            Equivalence classes of labels.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.labels)
        graph.add_edges_from(label_pairs)
        return map(list, nx.connected_components(graph))

    # --- Core Routines ---
    def get(self, label):
        """
        Maps a raw label to its class ID.

        Parameters
        ----------
        label : hashable
            Raw label (i.e. segment ID) to be mapped.

        Returns
        -------
        int
            Class ID corresponding to the label.
        """
        return self.mapping.get(label, 0) if self.labels else label

    # --- Helpers ---
    def node_labels(self, graph):
        """
        Gets the set of unique node labels from the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph from which to retrieve the node labels.

        Returns
        -------
        labels : Set[hashable]
            Labels corresponding to nodes in the graph identified by "key".
        """
        labels = graph.node_labels()
        if self.labels:
            labels = set().union(*(self.inverse_mapping[u] for u in labels))
        return labels
