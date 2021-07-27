Installation
------------

Prerequisites
~~~~~~~~~~~~~~

Cassiopeia currently requires python version 3.6, which is publicly available.

Cassiopeia needs to be downloaded from Github by cloning the directory onto your machine:

::

    git clone https://github.com/YosefLab/Cassiopeia.git

To run some of the models in Cassiopeia, you will also need to install `Gurobi <https://www.gurobi.com/>`_. Licenses are free to academic users and can be downloaded `here <https://www.gurobi.com/downloads/end-user-license-agreement-academic/>`_.


Installing
~~~~~~~~~~~

Once Cassiopeia is cloned into a directory onto your machine, enter into the directory with `cd Cassioepia`. To make installation simple, we have wrapped the installation steps into a MAKEFILE - this allows you to install Cassiopeia with the command:

::

    make install

To make sure that the package has been installed correctly, we recommend you also run all the unit tests with another command from the MAKEFILE:

::

    make test


