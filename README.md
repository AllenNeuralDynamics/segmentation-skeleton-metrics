# aind-segmentation-evaluation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

[![CI](https://github.com/AllenNeuralDynamics/library-repo-template/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenNeuralDynamics/library-repo-template/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/AllenNeuralDynamics/library-repo-template/branch/main/graph/badge.svg?token=ZVZ98GLA9V)](https://codecov.io/gh/AllenNeuralDynamics/library-repo-template)

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

Python package for performing a skeleton-based evaluation of a predicted segmentation of neuron axons. Given a predicted segmentation (i.e. pred_volume) and ground truth skeleton (i.e. target_graph), the evaluation is performed by detecting splits and merges in the prediction, then several statistics are computed from these quantities. 

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
