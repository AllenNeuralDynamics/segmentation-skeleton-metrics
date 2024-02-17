# segmentation-skeleton-metrics

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

Python package for performing a skeleton-based evaluation of a predicted segmentation of neural arbors. This tool detects topological mistakes (i.e. splits and merges) in the predicted segmentation by comparing the ground truth skeleton to it. Once this comparison is complete, several statistics (e.g. edge accuracy, split count, merge count) are computed and returned in a dictionary.


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
