"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import multiprocessing
from zipfile import ZipFile

from scipy.spatial import distance

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import swc_utils, utils


# -- projection utils --
def compute_run_length(zip_path, key, swc_ids):
    run_length = 0
    with ZipFile(zip_path, "r") as zip_file:
        for swc_id in swc_ids:
            content = utils.read_zip(zip_file, f"{swc_id}.swc").splitlines()
            graph = swc_utils.to_graph(content)
            run_length += gutils.compute_run_length(graph)
    return {key: run_length}


def compute_projections(kdtrees, zip_path):
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
            if cnt > 30:
                hits.add(key)
                break
    queue.put({process_id: hits})


# -- miscellaneous --
def dist(v_1, v_2):
    """
    Computes distance between "v_1" and "v_2".

    Parameters
    ----------
    v_1 : np.ndarray
        Vector.
    v_2 : np.ndarray
        Vector.

    Returns
    -------
    float
        Distance between "v_1" and "v_2".

    """
    return distance.euclidean(v_1, v_2)
