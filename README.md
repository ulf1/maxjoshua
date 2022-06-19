[![PyPI version](https://badge.fury.io/py/binsel.svg)](https://badge.fury.io/py/binsel)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ulf1/maxjoshua.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/maxjoshua/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ulf1/maxjoshua.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/maxjoshua/alerts/)

# maxjoshua
Feature selection for hard voting classifier and NN sparse weight initialization.

## Preface
I am naming this software package in memory of my late nephew Max Joshua Hamster (* 2005 to † June 18, 2022).

## Usage

### Forward Selection for Hard Voting Classifier
Load toy data set and convert features to binary.
```py
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
X = scale(load_breast_cancer().data, axis=0) > 0  # convert to binary features
y = load_breast_cancer().target
```

Select binary features
```py
import maxjoshua as mh
idx, neg, rho, results = mh.binsel(
    X, y, preselect=0.8, oob_score=True, subsample=0.5, 
    n_select=5, unique=True, n_draws=100, random_state=42)
```

### Forward Selection for Linear Regression
Load toy dataset.
```py
from sklearn.preprocessing import scale
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = scale(housing["data"], axis=0)
y = scale(housing["target"])
```

Select features
```py
import maxjoshua as mh
idx, loss, beta,  results = mh.fltsel(
    X, y, preselect=0.8, oob_score=True, subsample=0.5, 
    n_select=5, unique=True, n_draws=100, random_state=42)
```


### Initialize Sparse NN Layer



## Algorithm
The task is to select e.g. `n_select=3` features from a pool of many features.
These features might be the prediction of binary classifiers. 
The selected features are then combined into one hard-voting classifier.

A voting classifier should have the following properties

* each voter (a binary feature) should be highly correlated to the target variable
* the selected features should be uncorrelated.

The algorithm works as follows 

1. Generate multiple correlation matrices by bootstrapping. This includes `corr(X_i, X_j)` as well as `corr(Y, X_i)` computation. Also store the oob samples for evaluation.
2. For each correlation matrix do ...
    a. Preselect the `i*` with the highest `abs(corr(Y, X_i))` estimates (e.g. pick the `n_pre=?` highest absolute correlations)
    b. Slice a correlation matrix `corr(X_i*, X_j*)` and find the least correlated combination of `n_select=?` features. (see [`korr.mincorr`](https://github.com/kmedian/korr/blob/master/korr/mincorr.py))
    c. Compute the out-of-bag (OOB) performance (see step 1) of the hard-voter with the selected `n_select=?` features
3. Select the feature combination with the best OOB performance as final model.


## Appendix

### Installation
The `maxjoshua` [git repo](http://github.com/ulf1/maxjoshua) is available as [PyPi package](https://pypi.org/project/maxjoshua)

```sh
pip install maxjoshua
```

### Install a virtual environment

```sh
python3.7 -m venv .venv
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
* Run Unit Tests: `pytest`

Publish

```sh
pandoc README.md --from markdown --to rst -s -o README.rst
python setup.py sdist 
twine upload -r pypi dist/*
```

### Clean up 

```
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .venv
```

## Support
Please [open an issue](https://github.com/ulf1/maxjoshua/issues/new) for support.


## Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/ulf1/maxjoshua/compare/).
