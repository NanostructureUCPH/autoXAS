[![PyPI - Version](https://img.shields.io/pypi/v/autoXAS)](https://pypi.org/project/autoXAS/)
[![Read the Docs](https://img.shields.io/readthedocs/autoXAS)](https://autoxas.readthedocs.io/en/latest/)
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-10.26434/chemrxiv--2025--8pjq3-blue)](https://doi.org/10.26434/chemrxiv-2025-8pjq3)

![autoXAS_logo](https://github.com/UlrikFriisJensen/autoXAS/raw/main/figures/autoXAS_logo.svg)
# Automated analysis of X-ray Absorption Spectroscopy (XAS) data

1. [Installation](#installation)
    1. [Prerequisites](#prerequisites)
    2. [Install with pip](#install-with-pip)
    3. [Install locally](#install-locally)
2. [Usage](#using-autoxas)
3. [Cite](#cite)
4. [Issues and feature requests](#issues-and-feature-requests)
5. [Contributing to autoXAS](#contributing-to-autoxas)

## Installation

### Prerequisites

**autoXAS** requires python >= 3.7. 

If needed, create a new environment with a compatible python version:
```
conda create -n autoXAS_env python=3.10
```

```
conda activate autoXAS_env
```

### Install with pip

Run the following command to install the **autoXAS** package.
```
pip install autoXAS
```

### Install locally

Clone the repository.
```
git clone git@github.com:UlrikFriisJensen/autoXAS.git
```

Run the following command to install the **autoXAS** package.
```
pip install .
```

## Usage

See paper for example use.

## Cite

If you use the autoXAS package, please cite our paper:
```
@misc{friis-jensen_autoxas_2025,
	title = {{autoXAS}: {Automated} {Analysis} of {X}-ray {Absorption} {Spectroscopy} {Data}},
	shorttitle = {{autoXAS}},
	url = {https://chemrxiv.org/engage/chemrxiv/article-details/68c019983e708a7649f432de},
	doi = {10.26434/chemrxiv-2025-8pjq3},
	language = {en},
	publisher = {ChemRxiv},
	author = {Friis-Jensen, Ulrik and Jensen, Kirsten Marie Ørnsbjerg and Pittkowski, Rebecca},
	month = sep,
	year = {2025},
}
```

## Issues and feature requests

If you encounter any issues while using autoXAS, please report them by opening an issue on the GitHub repository.  Please provide as many details as possible, including steps to reproduce the issue and any error messages you received.

If you have any good ideas or features you're missing, then please suggest them as an Issue with the "enhancement" label on the GitHub repository.

## Contributing to autoXAS

If you would like to contribute to autoXAS please follow these steps:
1. Fork the repository
2. Make changes in a new branch
3. Submit a pull request

The changes will be reviewed and merged when they meet quality standards.
