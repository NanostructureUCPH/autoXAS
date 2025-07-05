Installation
=====

.. _installation:

To use autoXAS, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']


Prerequisites
-----------------

**autoXAS** requires python >= 3.7. 

If needed, create a new environment with a compatible python version:
.. code-block:: console
    conda create -n autoXAS_env python=3.10

.. code-block:: console
    conda activate autoXAS_env

Install with pip
-----------------

Run the following command to install the **autoXAS** package.
.. code-block:: console
    pip install autoXAS

Install locally
-----------------

Clone the repository.
.. code-block:: console
    git clone git@github.com:UlrikFriisJensen/autoXAS.git

Run the following command to install the **autoXAS** package.
.. code-block:: console
    pip install .