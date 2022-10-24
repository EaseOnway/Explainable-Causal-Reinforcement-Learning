.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*A Fast Conditional Independence Test (FCIT).*

Introduction
-----------
Let *x, y, z* be random variables. Then deciding whether *P(y | x, z) = P(y | z)* 
can be difficult, especially if the variables are continuous. This package 
implements a simple yet efficient and effective conditional independence test,
described in [link to arXiv when we write it up!]. Important features that differentiate
this test from competition:

* It is fast. Worst-case speed scales as O(n_data * log(n_data) * dim), where dim is max(x_dim + z_dim, y_dim). However, amortized speed is O(n_data * log(n_data) * log(dim)).

* It applies to cases where some of x, y, z are continuous and some are discrete, or categorical (one-hot-encoded).

* It is very simple to understand and modify.

* It can be used for unconditional independence testing with almost no changes to the procedure.

We have applied this test to tens of thousands of samples of thousand-dimensional datapoints in seconds. For smaller dimensionalities and sample sizes, it takes a fraction of a second. The algorithm is described in [arXiv link coming], where we also provide detailed experimental results and comparison with other methods. However for now, you should be able to just look through the code to understand what's going on -- it's only 90 lines of Python, including detailed comments!

Usage
-----
Basic usage is simple, and the default settings should work in most cases. To perform an *unconditional test*, use dtit.test(x, y):

.. code:: python

  import numpy as np
  from fcit import fcit
  
  x = np.random.rand(1000, 1)
  y = np.random.randn(1000, 1)
  
  pval_i = fcit.test(x, y) # p-value should be uniform on [0, 1].
  pval_d = fcit.test(x, x + y) # p-value should be very small.
  
To perform a conditional test, just add the third variable z to the inputs:
 
.. code:: python

  import numpy as np
  from fcit import fcit
  
  # Generate some data such that x is indpendent of y given z.
  n_samples = 1000
  z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
  x = np.vstack([np.random.multinomial(20, p) for p in z]).astype(float)
  y = np.vstack([np.random.multinomial(20, p) for p in z]).astype(float)
  
  # Check that x and y are dependent (p-value should be uniform on [0, 1]).
  pval_d = fcit.test(x, y)
  # Check that z d-separates x and y (the p-value should be small).
  pval_i = fcit.test(x, y, z)

Installation
-----------
pip install fcit


Requirements
------------
Tested with Python 3.6 and

    * joblib >= 0.11
    * numpy >= 1.12
    * scikit-learn >= 0.18.1
    * scipy >= 0.16.1

.. _pip: http://www.pip-installer.org/en/latest/
