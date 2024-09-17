# SkeletonMetrics

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

Python package for evaluating the accuracy of a predicted segmentation of neuronal arbors by comparing it to a set of ground truth skeletons. This tool detects topological mistakes such as splits and merges in the predicted segmentation, then computes several skeleton-based metrics that quantify its topological accuracy.

## Details

Ground truth skeletons must be stored as SWC files so that each can be represented as a graph with xyz coordinates as a node-level attribute. The evaluation is performed by first labeling the nodes of ground truth skeletons with the corresponding segment ids from the predicted segmentation. Topological mistakes are then detected by examining the labels of individual nodes, neighboring nodes, and nodes across different ground truth skeletons.

![Edges in skeletons are either correctly or incorrectly reconstructed based on the presence of mergers or splits that affect nodes attached to an edge. Colors correspond to segment IDs. From top to bottom: correct edge (both nodes have the same ID), split edge
(nodes assigned to different segments), omitted edge (one or two nodes do not have an associated ID), merged edge (node assigned to a segment that covers more than one skeleton).
](imgs/topological_mistakes.png)



Metrics computed for each ground truth skeleton:

- Number of Splits: Number of segments that a ground truth skeleton is broken into.
- Number of Merges: Number multiple segments are incorrectly merged into a single segment.
- Percentage of Omit Edges: Proportion of edges in the ground truth that are omitted in the predicted segmentation.
- Percentage of Merged Edges: Proportion of edges that are merged in the predicted segmentation compared to the ground truth.
- Edge Accuracy: Evaluates how accurately the edges of the predicted segmentation match the ground truth.
- Expected Run Length (ERL): Expected length of segments or edges in the predicted segmentation.

## Usage

Here is a simple example of evaluating a predicted segmentation. Note that this package supports a number of different input types, see documentation for details. 

```python
import os

from aind_segmentation_evaluation.evaluate import run_evaluation
from aind_segmentation_evaluation.conversions import volume_to_graph
from tifffile import imread


if __name__ == "__main__":

    # Initializations
    data_dir = "./resources"
    target_graphs_dir = os.path.join(data_dir, "target_graphs")
    path_to_target_labels = os.path.join(data_dir, "target_labels.tif")
    pred_labels = imread(os.path.join(data_dir, "pred_labels.tif"))
    pred_graphs = volume_to_graph(pred_labels)

    # Evaluation
    stats = run_evaluation(
        target_graphs_dir,
        path_to_target_labels,
        pred_graphs,
        pred_labels,
        filetype="tif",
        output="tif",
        output_dir=data_dir,
        permute=[2, 1, 0],
        scale=[1.101, 1.101, 1.101],
    )

    # Write out results
    print("Graph-based evaluation...")
    for key in stats.keys():
        print("   {}: {}".format(key, stats[key])

```

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

To install this package from PyPI, run
```bash
pip install aind-segmentation-evaluation
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
