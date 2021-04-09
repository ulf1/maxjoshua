[![PyPI version](https://badge.fury.io/py/binsel.svg)](https://badge.fury.io/py/binsel)

# binsel
Feature selection for Hard Voting classifier.


## Usage
Check the [`binsel_hardvote` example](https://github.com/kmedian/binsel/blob/master/examples/binsel_hardvote.ipynb) folder for notebooks.


## Algorithm
The task is to select e.g. `n_select=3` binary features from a pool of many binary features.
These binary features might be the prediction of binary classifiers. 
The selected binary features are then combined into one hard-voting classifier.

A voting classifier should have the following properties

* each voter (a binary feature) should be highly correlated to the target variable
* the selected binary features should be uncorrelated.

The algorithm works as follows 

1. Generate multiple correlation matrices by bootstrapping (see [`korr.bootcorr`](https://github.com/kmedian/korr/blob/master/korr/bootcorr.py)). This includes `corr(X_i, X_j)` as well as `corr(Y, X_i)` computation. Also store the oob samples for evaluation.
2. For each correlation matrix do ...
    a. Preselect the `i*` with the highest `abs(corr(Y, X_i))` estimates (e.g. pick the `n_pre=?` highest absolute correlations)
    b. Slice a correlation matrix `corr(X_i*, X_j*)` and find the least correlated combination of `n_select=?` features. (see [`korr.mincorr`](https://github.com/kmedian/korr/blob/master/korr/mincorr.py))
    c. Compute the out-of-bag (OOB) performance (see step 1) of the hard-voter with the selected `n_select=?` binary features
3. Select the binary feature combination with the best OOB performance as final model.


## Appendix

### Installation
The `binsel` [git repo](http://github.com/kmedian/binsel) is available as [PyPi package](https://pypi.org/project/binsel)

```sh
pip install binsel
```

### Install a virtual environment

```sh
python3.6 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-demo.txt
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `python -W ignore -m unittest discover`
* Upload to PyPi with twine: `python setup.py sdist && twine upload -r pypi dist/*`

### Clean up 

```
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .venv
```

## Support
Please [open an issue](https://github.com/kmedian/binsel/issues/new) for support.


## Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/kmedian/binsel/compare/).
