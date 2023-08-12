[![PyPI version](https://badge.fury.io/py/maxjoshua.svg)](https://badge.fury.io/py/maxjoshua)

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

Select binary features. Each row in the `results` list contains the `n_select` column indices of `X`, the notice if the binary features were negated, and the sum of absolute MCC correlation coeffcients between the selected features.
```py
import maxjoshua as mh
idx, neg, rho, results = mh.binsel(
    X, y, preselect=0.8, oob_score=True, subsample=0.5, 
    n_select=5, unique=True, n_draws=100, random_state=42)
```

**Algorithm**. 
The task is to select e.g. `n_select` features from a pool of many features.
These features might be the prediction of binary classifiers. 
The selected features are then combined into one hard-voting classifier.

A voting classifier should have the following properties

* each voter (a binary feature) should be highly correlated to the target variable
* the selected features should be uncorrelated.

The algorithm works as follows 

1. Generate multiple correlation matrices by bootstrapping. This includes `corr(X_i, X_j)` as well as `corr(Y, X_i)` computation. Also store the oob samples for evaluation.
2. For each correlation matrix do ...
    a. Preselect the `i*` with the highest `abs(corr(Y, X_i))` estimates (e.g. pick the `n_pre=?` highest absolute correlations)
    b. Slice a correlation matrix `corr(X_i*, X_j*)` and find the least correlated combination of `n_select` features. (see [`korr.mincorr`](https://github.com/kmedian/korr/blob/master/korr/mincorr.py))
    c. Compute the out-of-bag (OOB) performance (see step 1) of the hard-voter with the selected `n_select=?` features
3. Select the feature combination with the best OOB performance as final model.


### Forward Selection for Linear Regression
Load toy dataset.
```py
from sklearn.preprocessing import scale
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = scale(housing["data"], axis=0)
y = scale(housing["target"])
```

Select real-numbered features. Each row in the `results` list contains the `n_select` column indices of `X`, the ridge regression coefficents `beta` and the RMSE `loss`.
Warning! Please note that the features `X` and the target `y` must be scaled because `mh.fltsel` uses an L2-penalty on `beta` coefficients, and doesn't used an intercept term to shift `y`.
```py
import maxjoshua as mh
from sklearn.preprocessing import scale

idx, beta, loss, results = mh.fltsel(
    scale(X), scale(y), preselect=0.8, oob_score=True, subsample=0.5, 
    n_select=5, unique=True, n_draws=100, random_state=42, l2=0.01)
```


### Initialize Sparse NN Layer
The idea is to run `mh.fltsel` to generate an ensemble of linear models, and combine them in a sparse linear neural network layer, i.e., the number of output neurons is the ensemble size.
In case of small datasets, the sparse NN layer is non-trainable because because each submodel was already estimated and selected with two-way data splits in `mh.fltsel` (see `oob_scores` and `subsample`). 
The sparse NN layers basically produces submodel predictions for meta model in the next layer, i.e., a simple dense linear layer.
The inputs of the sparse NN layer must be normalized for which a layer normalization layers is trained.

```py
import maxjoshua as mh
import tensorflow as tf
import sklearn.preprocessing

# create toy dataset
import sklearn.datasets
X, y = sklearn.datasets.make_regression(
    n_samples=1000, n_features=100, n_informative=20, n_targets=3)

# feature selection
# - always scale the inputs and targets -
indices, values, num_in, num_out = mh.pretrain_submodels(
    sklearn.preprocessing.scale(X), 
    sklearn.preprocessing.scale(y), 
    num_out=64, n_select=3)

# specify the model
model = tf.keras.models.Sequential([
    # sub-models
    mh.SparseLayerAsEnsemble(
        num_in=num_in, 
        num_out=num_out, 
        sp_indices=indices, 
        sp_values=values,
        sp_trainable=False,
        norm_trainable=True,
    ),
    # meta model
    tf.keras.layers.Dense(
        units=3, use_bias=False,
        # kernel_constraint=tf.keras.constraints.NonNeg()
    ),
    # scale up
    mh.InverseTransformer(
        units=3,
        init_bias=y.mean(), 
        init_scale=y.std()
    )
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4, beta_1=.9, beta_2=.999, epsilon=1e-7, amsgrad=True),
    loss='mean_squared_error'
)

# train
history = model.fit(X, y, epochs=3)
```




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
