"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that runs of demo of using this library to compute skeleton metrics.

"""

import numpy as np
from xlwt import Workbook

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
    print("# splits:", np.sum(list(skeleton_metric.split_cnt.values())))
    print("# merges:", np.sum(list(skeleton_metric.merge_cnt.values())))

    # Save results
    path = f"{output_dir}/evaluation_results.xls"
    save_results(path, full_results)


def save_results(path, stats):
    # Initialize
    wb = Workbook()
    sheet = wb.add_sheet("Results")
    sheet.write(0, 0, "swc_id")

    # Label rows and columns
    swc_ids = list(stats.keys())
    for i, swc_id in enumerate(swc_ids):
        sheet.write(i + 1, 0, swc_id)

    metrics = list(stats[swc_id].keys())
    for i, metric in enumerate(metrics):
        sheet.write(0, i + 1, metric)

    # Write stats
    for i, swc_id in enumerate(swc_ids):
        for j, metric in enumerate(metrics):
            sheet.write(i + 1, j + 1, round(stats[swc_id][metric], 4))

    wb.save(path)


if __name__ == "__main__":
    # Initializations
    output_dir = "./"
    pred_labels_path = "./pred_labels.tif"
    fragments_pointer = "./pred_swcs.zip"
    groundtruth_pointer = "./target_swcs.zip"

    # Run
    evaluate()
