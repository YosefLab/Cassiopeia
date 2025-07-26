Installation
------------

Prerequisites
~~~~~~~~~~~~~~

Cassiopeia currently requires python version 3.10 and above, which are publicly available.

Installing
~~~~~~~~~~~

For basic users, you can download Cassiopeia via pip as so:

::
    pip install git+https://github.com/YosefLab/Cassiopeia@master#egg=cassiopeia-lineage

To run some of the models in Cassiopeia, you will also need to install `Gurobi <https://www.gurobi.com/>`_. Licenses are free to academic users and can be downloaded `here <https://www.gurobi.com/downloads/end-user-license-agreement-academic/>`_.

For developers, you can clone the code from Github as so:

::

    git clone https://github.com/YosefLab/Cassiopeia.git

Once Cassiopeia is cloned into a directory onto your machine, enter into the directory with `cd Cassiopeia`. To make installation simple, we have wrapped the installation steps into a MAKEFILE - this allows you to install Cassiopeia with the command:

::

    make install

To make sure that the package has been installed correctly, we recommend you also run all the unit tests with another command from the MAKEFILE:

::

    make test


