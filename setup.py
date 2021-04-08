from setuptools import setup
import pypandoc


setup(name='binsel',
      version='0.2.2',
      description='Feature selection for Hard Voting classifier',
      long_description=pypandoc.convert('README.md', 'rst'),
      url='http://github.com/kmedian/binsel',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['binsel'],
      install_requires=[
          'setuptools>=40.0.0',
          'numpy>=1.14.5',
          'korr>=0.8.2',
          'scikit-learn>=0.20.0'
      ],
      python_requires='>=3.6',
      zip_safe=True)
