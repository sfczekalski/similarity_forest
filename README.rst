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

Below you can find neccessary steps to install the project package. 
It assumes that you have Anaconda (or Miniconda) installed.
If you don't, follow the steps from the docs: 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/ 

Package instalation
-------------------
1. Make sure that conda it's up to date ::

    conda update conda

2. Clone this repository ::

    git clone https://github.com/sfczekalski/similarity_forest

3. Go to the project folder ::

    cd similarity_forest

3. Create conda environment ::

    conda env create --file environment.yml

4. Activate conda environment ::

    conda activate similarity-forest

5. Install simforest package ::

    pip install .
