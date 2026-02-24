

This is the package for single-layer combinatorial fusion analysis (CFA).

The input is a pandas DataFrame where each column contains a model’s output (usually probabilities), along with a binary target vector (0/1) and a performance metric. Currently, accuracy, AUROC and precision@k are supported.

Then you call cfa_single_layer function from the package, you will get the fusion results right away.

The packgae also provides functions to draw rank-score function graph and performance plot for the fusion results. 

For how to use the packgae, please see test_cfa.ipynb file.

This project is still ongoing. Multilayer combinatorial fusion is on the way. Stay tuned.

## Installation (from GitHub)

### Requirements
- Python: **>= 3.9**
- Tested dependency ranges:
  - numpy: **>=1.22,<3**
  - pandas: **>=1.5,<3**
  - scikit-learn: **>=1.1,<2**
  - matplotlib: **>=3.6,<4**
  - seaborn: **>=0.12,<0.14**


##  Install directly from GitHub
```bash
pip install "git+https://github.com/ywu60/Combinatorial-Fusion-Analysis.git"
