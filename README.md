[![Build Status](https://travis-ci.org/kmedian/binsel.svg?branch=master)](https://travis-ci.org/kmedian/binsel)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/kmedian/binsel/master?urlpath=lab)

# binsel
Feature selection for Hard Voting classifier.


## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Commands](#commands)
* [Support](#support)
* [Contributing](#contributing)


## Installation
The `binsel` [git repo](http://github.com/kmedian/binsel) is available as [PyPi package](https://pypi.org/project/binsel)

```
pip install binsel
```


## Usage
Check the [examples](http://github.com/kmedian/binsel/examples) folder for notebooks.


## Commands
* Check syntax: `flake8 --ignore=F401`
* Run Unit Tests: `python -W ignore -m unittest discover`
* Remove `.pyc` files: `find . -type f -name "*.pyc" | xargs rm`
* Remove `__pycache__` folders: `find . -type d -name "__pycache__" | xargs rm -rf`
* Upload to PyPi with twine: `python setup.py sdist && twine upload -r pypi dist/*`


## Support
Please [open an issue](https://github.com/kmedian/binsel/issues/new) for support.


## Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/kmedian/binsel/compare/).
