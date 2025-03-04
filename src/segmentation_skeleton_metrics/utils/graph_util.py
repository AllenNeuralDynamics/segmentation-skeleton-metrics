"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Code for building a custom graph object called a SkeletonGraph and helper
routines for working with graph.

"""
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm

import networkx as nx

from segmentation_skeleton_metrics.skeleton_graph import SkeletonGraph
from segmentation_skeleton_metrics.utils import swc_util, util


class GraphBuilder:
    """
    A class that builds graphs constructed from SWC files.

    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        label_mask=None,
        selected_ids=None,
        use_anisotropy=True,
    ):
        """
        Instantiates a GraphBuilder object.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        label_mask : ImageReader, optional
            Predicted segmentation mask.
        selected_ids : Set[int], optional
            Only SWC files with an swc_id contained in this set are read. The
            default is None.
        use_anisotropy : bool, optional
            Indication of whether to apply anisotropy to coordinates in SWC
            files. The default is True.

        Returns
        -------
        None

        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.label_mask = label_mask

        # Reader
        anisotropy = anisotropy if use_anisotropy else (1.0, 1.0, 1.0)
        self.swc_reader = swc_util.Reader(
            anisotropy, selected_ids=selected_ids
        )

    def run(self, swc_pointer):
        """
        Builds graphs by reading SWC files to extract content which is then
        loaded into a custom SkeletonGraph object. Optionally, the nodes are
        labeled if a "label_mask" is provided.

        Parameters
        ----------
        swc_pointer : Any
            Object that points to SWC files to be read.

        Returns
        -------
        dict
            A dictionary where the keys are unique identifiers (i.e. filenames
            of SWC files) and values are the correspondign SkeletonGraph.

        """
        graphs = self._build_graphs_from_swcs(swc_pointer)
        graphs = self._label_graphs_with_segmentation(graphs)
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
        dict
            A dictionary where the keys are unique identifiers (i.e. filenames
            of SWC files) and values are the correspondign SkeletonGraph.

        """
        # Initializations
        swc_dicts = self.swc_reader.read(swc_pointer)
        pbar = tqdm(total=len(swc_dicts), desc="Build Graphs")

        # Main
        graphs = dict()
        if len(swc_dicts) > 10 ** 4:
            while len(swc_dicts) > 0:
                swc_dict = swc_dicts.pop()
                graphs.update(self.to_graph(swc_dict))
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
        SkeletonGraph
            Graph built from an SWC file.

        """
        # Initialize graph
        graph = SkeletonGraph(anisotropy=self.anisotropy)
        graph.init_voxels(swc_dict["voxel"])
        graph.set_filename(swc_dict["swc_id"] + ".swc")
        graph.set_nodes(len(swc_dict["id"]))

        # Build graph
        id_lookup = dict()
        for i, id_i in enumerate(swc_dict["id"]):
            id_lookup[id_i] = i
            if swc_dict["pid"][i] != -1:
                parent = id_lookup[swc_dict["pid"][i]]
                graph.add_edge(i, parent)
                graph.run_length += graph.dist(i, parent)

        # Set graph-level attributes
        graph.graph["n_edges"] = graph.number_of_edges()
        return {swc_dict["swc_id"]: graph}

    # --- Label Graphs ---
    def _label_graphs_with_segmentation(self, graphs):
        return graphs

    def _label_graph(self, key):
        pass


class LabelHandler:
    def __init__(self, connections_path=None, valid_labels=None):
        """
        Instantiates a LabelHandler object and builds the label mappings if a
        connections path is provided.

        Parameters
        ----------
        connections_path : str, optional
            Path to file containing pairs of segment IDs that were merged. The
            default is None.
        valid_labels : Set[int], optional
            Segment IDs that can be assigned to nodes. This argument accounts
            for segments that were been removed due to some type of filtering.
            The default is None.

        Returns
        -------
        None

        """
        self.mapping = dict()  # Maps label to equivalent class id
        self.inverse_mapping = dict()  # Maps class id to list of labels
        self.processed_labels = set()
        self.valid_labels = valid_labels or set()
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

        Returns
        -------
        None

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
        Builds a graph from a list of labels and connection data. The nodes
        are initialized with "self.valid_labels", then edges are added between
        nodes based on a list of connections specified in a file.

        Parameters
        ----------
        connections_path : str
            Path to a text file containing connections. Each line represents a
            connection between two segmentation ids.

        Returns
        -------
        networkx.Graph
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

    # --- Main ---
    def get(self, label):
        if self.use_mapping():
            return self.mapping.get(label, 0)
        elif self.valid_labels:
            return 0 if label not in self.valid_labels else label
        return label

    def get_class(self, label):
        return self.inverse_mapping[label] if self.use_mapping() else [label]

    def use_mapping(self):
        return True if len(self.mapping) > 0 else False


# -- Miscellaneous --
def count_splits(graph):
    """
    Counts the number of splits in "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be evaluated.

    Returns
    -------
    int
        Number of splits in "graph".

    """
    return max(nx.number_connected_components(graph) - 1, 0)
