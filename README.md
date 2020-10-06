.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/simforest/badge/?version=latest
.. _ReadTheDocs: https://simforest.readthedocs.io/en/latest/?badge=latest

Similarity Forest : A similarity-based decision tree ensemble
=============================================================

This is my MS thesis project conducted at Poznan University of Technology under the supervision of
prof. Mikołaj Morzy
It aims to extend the work described in Similarity Forest by Sathe and Aggarwal and provide ease to use,
Scikit-Learn compatible implementation.
It can be used in Scikit-Learn pipelines and (hyper)parameter search, it includes testing (API compliance) and more.
I will include necessary documentation.

To recreate the environment and install the package, follow these steps:
1. If you have Anaconda (or Miniconda) installed, make sure it's up to date
```bash
conda update conda
```
If not, install it by following the instruction in the documentation: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
2. Clone this repository
```bash
git clone https://github.com/sfczekalski/similarity_forest
```
3. Create conda environment
```bash
conda create --name similarity-forest --file environment.yml
```
4. Activate conda environment
```bash
conda activate similarity-forest
```
5. Install simforest package
```bash
pip install .
```