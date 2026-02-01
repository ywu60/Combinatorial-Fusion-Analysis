

This is the package for single-layer combinatorial fusion analysis (CFA).

The input is a pandas DataFrame where each column contains a model’s output (usually probabilities), along with a binary target vector (0/1) and a performance metric. Currently, accuracy and AUROC are supported.

Then you call cfa_single_layer function from the package, you will get the fusion results right away. Normally it's less than 3 seconds.

The packgae also provides functions to draw rank-score function graph and performance plot for the fusion results. 

For how to use the packgae, please see test_cfa.ipynb file.

This project is still ongoing. Multilayer combinatorial fusion is on the way. Stay tuned.
