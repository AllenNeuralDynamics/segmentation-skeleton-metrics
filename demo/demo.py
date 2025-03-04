"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that runs of demo of using this library to compute skeleton metrics.

"""

from segmentation_skeleton_metrics.utils import util
from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.utils.img_util import TiffReader


def evaluate():
    """
    Evaluates the accuracy of a predicted segmentation by comparing it to a
    set of ground truth skeletons, then reports and saves various performance
    metrics.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # Initializations
    pred_labels = TiffReader(pred_labels_path)
    skeleton_metric = SkeletonMetric(
        groundtruth_pointer,
        pred_labels,
        fragments_pointer=fragments_pointer,
        output_dir=output_dir,
    )
    full_results, avg_results = skeleton_metric.run()

    # Report results
    print(f"\nAveraged Results...")
    for key in avg_results.keys():
        print(f"   {key}: {round(avg_results[key], 4)}")

    print(f"\nTotal Results...")
    print("# splits:", skeleton_metric.count_total_splits())
    print("# merges:", skeleton_metric.count_total_merges())

    # Save results
    path = f"{output_dir}/evaluation_results.xls"
    util.save_results(path, full_results)


if __name__ == "__main__":
    # Initializations
    output_dir = "./"
    pred_labels_path = "./pred_labels.tif"
    fragments_pointer = "./pred_swcs.zip"
    groundtruth_pointer = "./target_swcs.zip"

    # Run
    evaluate()
