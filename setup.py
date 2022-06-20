from setuptools import setup
import os


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fp:
        s = fp.read()
    return s


def get_version(path):
    with open(path, "r") as fp:
        lines = fp.read()
    for line in lines.split("\n"):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(name='maxjoshua',
      version=get_version("maxjoshua/__init__.py"),
      description=(
          "Feature selection for hard voting classifier and "
          "NN sparse weight initialization."
      ),
      long_description=read('README.rst'),
      url='http://github.com/ulf1/maxjoshua',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='Apache License 2.0',
      packages=['maxjoshua'],
      install_requires=[
          'numpy>=1.14.5,<2',
          'korr>=0.10.0,<1',
          'numba>=0.55.2,<1',
          'numpy_linreg>=0.1.2,<1',
          'tensorflow>=2.9.0,<3',
          'keras-tweaks>=0.2.2,<1',
          'scikit-learn>=0.20.0'
      ],
      python_requires='>=3.7',
      zip_safe=True)
