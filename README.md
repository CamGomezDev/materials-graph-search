# Materials Graph Search

This repository contains some functions and utilities useful for turning a set of materials into a graph, each node representing each material, and then for searching within that graph possible new materials of a certain type by judging how similar they are to others. You can see a series of examples in the [examples](../examples.ipynb) file.

All of the functions, properly documented, is inside of the [graph_utilities.py](../graph_utilities.py) file.


# Installation

For all of the functions in this repository to run it is necessary to have installed the following dependencies:
`numpy`, `pandas`, `tqdm`, `scipy`, `fa2`, `matplotlib`, `plotly`, `sklearn`, `umap`, `numba`