# jungle - Phylogenetic analysis in Python

Jungle is a toolkit for analyzing phylogenetic signatures of evolution.


## What is jungle good for?

- Calculating classical signatures of selection, such as the site frequency spectrum or Tajima's D.

- Simulating trees under coalescent models, such as Kingman's coalescent.

- Calculating node-level fitness scores.


## Getting started

### Setting up environment

Jungle uses a Python 2.7 environment. To create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml -n jungle
```

To activate the environment:

```bash
conda activate jungle
```

### Installing jungle

To install jungle, clone this repository to a local directory.

To import jungle, append the jungle directory to your path, then do the import:

```python
import sys
sys.path.append("/path/to/jungle")
import jungle as jg
```

## Using jungle

Examples of analysis are in `examples` as Jupyter notebooks.

## Contact
If you have questions or comments, please contact Felix Horns at <rfhorns@gmail.com>.

## Disclaimer
This project is not maintained. Software is provided as is and requests for support may not be addressed.
