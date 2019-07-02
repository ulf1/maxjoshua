from setuptools import setup


def read(fname):
    import os
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='binsel',
      version='0.1.1',
      description='Feature selection for Hard Voting classifier',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='http://github.com/kmedian/binsel',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['binsel'],
      install_requires=[
          'setuptools>=40.0.0',
          'nose>=1.3.7',
          'korr>=0.8.1'
      ],
      python_requires='>=3.6',
      zip_safe=False)
