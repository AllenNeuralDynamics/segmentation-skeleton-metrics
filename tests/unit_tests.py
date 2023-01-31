import os
import numpy as np
import aind_segmentation_evaluation.graph_routines as gr
import aind_segmentation_evaluation.split_metric as sp
import aind_segmentation_evaluation.merge_metric as me
import networkx as nx
from numpy.random import random


def make_graph(num_nodes=10):
    graph = nx.Graph()
    for i in range(num_nodes):
        graph.add_node(i, xyz=(i, i, 0), idx=(i, i, 0))
        if i != 0:
            graph.add_edge(i - 1, i)
    return graph

def make_volume(shape=(10,10,10)):
    volume = np.zeros(shape, dtype=np.uint)
    for i in range(shape[0]):
        volume[i, i, 0] = 1
    return volume

def test_break(graph, volume):
    corrupt_volume = volume.copy()
    corrupt_volume[0, 0, 0] = 0
    corrupt_volume[1, 1, 0] = 0
    split_metric = sp.SplitMetric(
        shape,
        target_graphs=graph,
        pred_volume=corrupt_volume,
    )
    split_metric.detect_mistakes()
    test1 = split_metric.site_cnt == 0
    test2 = split_metric.edge_cnt == 2
    result = 'Pass' if test1 and test2 else 'Fail'
    return result

def test_simple_split(graph, volume):
    split_volume = make_split(volume)
    split_metric = sp.SplitMetric(
        shape,
        target_graphs=graph,
        pred_volume=split_volume,
    )
    split_metric.detect_mistakes()
    test1 = split_metric.site_cnt == 1
    test2 = split_metric.edge_cnt == 1
    result = 'Pass' if test1 and test2 else 'Fail'
    return result

def test_complex_split(graph, volume):
    split_volume = make_split(volume, num_split_edges=3)
    split_metric = sp.SplitMetric(
        shape,
        target_graphs=graph,
        pred_volume=split_volume,
    )
    split_metric.detect_mistakes()
    test1 = split_metric.site_cnt == 1
    test2 = split_metric.edge_cnt == 3
    result = 'Pass' if test1 and test2 else 'Fail'
    return result

def test_simple_merge(graph, volume):
    split_volume = make_split(volume)
    merge_metric = me.MergeMetric(
        shape,
        pred_graphs=graph,
        target_volume=split_volume
    )
    merge_metric.detect_mistakes()
    test1 = merge_metric.site_cnt == 1
    test2 = merge_metric.edge_cnt == 5
    result = 'Pass' if test1 and test2 else 'Fail'
    return result

def test_complex_merge(graph, volume):
    split_volume = make_split(volume, num_split_edges=2)
    print(split_volume[:,:,0])
    merge_metric = me.MergeMetric(
        shape,
        pred_graphs=graph,
        target_volume=split_volume
    )
    merge_metric.detect_mistakes()
    test1 = merge_metric.site_cnt == 1
    test2 = merge_metric.edge_cnt == 4
    print('edge_cnt:', merge_metric.edge_cnt)
    print('site_cnt:', merge_metric.site_cnt)
    result = 'Pass' if test1 and test2 else 'Fail'
    return result

def make_split(volume, num_split_edges=1):
    # Create split
    for i in range(5, 5 + num_split_edges - 1):
        volume[i, i, 0] = 0

    # Change labels
    for i in range(5 + num_split_edges - 1, 10):
        volume[i, i, 0] = 2
    return volume


if __name__ == "__main__":
    print("Running unit tests...")

    # Create test case
    shape = (15, 15, 15)
    graph = [make_graph()]
    volume = make_volume()

    # Split detection
    break_result = test_break(graph, volume)
    simple_split_result = test_simple_split(graph, volume)
    complex_split_result = test_complex_split(graph, volume)

    simple_merge_result = test_simple_merge(graph, volume)
    complex_merge_result = test_complex_merge(graph, volume)

    # Print out results
    print('   Result of break test:', break_result)
    print('   Result of simple split test:', simple_split_result)
    print('   Result of simple complex split test:', complex_split_result)
    print('')

    print('   Result of simple merge test:', simple_merge_result)
    print('   Result of complex merge test:', complex_merge_result)
