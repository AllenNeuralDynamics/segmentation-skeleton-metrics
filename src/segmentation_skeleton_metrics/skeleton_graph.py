import networkx as nx
import numpy as np


class SkeletonGraph(nx.Graph):

    def __init__(self):
        # Call parent class
        super(SkeletonGraph, self).__init__()

    def get_labels(self):
        return np.unique(self.graph["label"])

    def nodes_with_label(self, label):
        return np.where(self.graph["label"] == label)[0]
