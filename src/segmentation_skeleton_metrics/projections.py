"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import multiprocessing
import os
from zipfile import ZipFile

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import swc_utils, utils


# -- projection utils --
def compute_run_length(projections, graphs, inv_label_map):
    run_length = 0
    for key in projections.keys():
        if inv_label_map:
            run_length += rl_with_label_map(graphs, inv_label_map, key)
        elif key in graphs.keys():
            run_length += gutils.compute_run_length(graphs[key])
    return run_length


def rl_with_label_map(graphs, inv_label_map, key):
    run_length = 0
    if key in inv_label_map.keys():
        for swc_id in inv_label_map[key]:
            if swc_id in graphs.keys():
                run_length += gutils.compute_run_length(graphs[swc_id])
    return run_length


def compute_projections(graph, key):
    # Main
    projections = dict()
    for i in graph.nodes:
        label = graph.nodes[i]["label"]
        if label in projections.keys():
            projections[label] += 1
        else:
            projections[label] = 0

    # Finish
    keys = list()
    for key, cnt in projections.items():
        if cnt < 30:
            keys.append(key)
    return {key: utils.delete_keys(projections, keys)}


def compute_run_length_old(zip_path, key, swc_ids, output_dir=None):
    # Initializations
    anisotropy = [1.0 / 0.748, 1.0 / 0.748, 1.0]  # hard coded
    run_length = 0
    if output_dir:
        swc_dir = os.path.join(output_dir, key)
        utils.mkdir(swc_dir, delete=True)

    # Main
    with ZipFile(zip_path, "r") as zip_file:
        for swc_id in swc_ids:
            content = utils.read_zip(zip_file, f"{swc_id}.swc").splitlines()
            graph = swc_utils.to_graph(content, anisotropy=anisotropy)
            run_length += gutils.compute_run_length(graph)
            if output_dir:
                pass  # save graph as swc
    return key, run_length


def compute_projections_old(kdtrees, zip_path):
    # Initializations
    manager = multiprocessing.Manager()
    coords = swc_utils.parse_local_zip(zip_path, 0, [1, 1, 1])
    shared_coords = manager.dict(coords)
    del coords

    # Main
    processes = []
    queue = multiprocessing.Queue()
    for key, kdtree in kdtrees.items():
        process = multiprocessing.Process(
            target=query_kdtree, args=(kdtree, key, shared_coords, queue)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    return utils.merge_dict_list([queue.get() for _ in kdtrees])


def query_kdtree(kdtree, process_id, coords, queue):
    """
    Queries a k-d tree to find coordinates that have a sufficient number of
    nearby points within a distance threshold.

    Parameters
    ----------
    kdtree : scipy.spatial.KDTree
        KD-Tree object used to query the coordinates.
    process_id : str
        Identifier for the process executing the query.
    coords : dict
        Dictionary that maps a fragment id to an array containing the xyz that
        comprise that fragment.
    queue : multiprocessing.Queue
        Queue to put the results into.

    Returns
    -------
    None

    """
    hits = set()
    for key, arr in coords.items():
        cnt = 0
        for xyz in arr:
            d, _ = kdtree.query(xyz, k=1)
            cnt += 1 if d < 3 else 0
            if cnt > 25:
                hits.add(key)
                break
    queue.put({process_id: hits})
