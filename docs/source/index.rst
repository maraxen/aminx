Aminx Documentation
=========================

.. image:: https://img.shields.io/badge/coverage-90%25-brightgreen.svg
  :target: https://github.com/yourusername/Aminx/actions/workflows/pytest.yml
  :alt: Test coverage

`Run on Colab <https://colab.research.google.com/github/maraxen/Aminx/blob/main/examples/colab_inference_demo.ipynb>`_

Welcome to Aminx's documentation! This project provides a functional interface for ProteinMPNN, leveraging the JAX ecosystem for accelerated computation.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  ../COMPOSITION_GUIDE
  ../MODEL_FAMILIES
  api/index
  tutorials/index
  examples/index

Key Features
------------

* **Increased Transparency:** Clear and functional interface for ProteinMPNN
* **JAX Compatibility:** Efficient computation with JAX's functional programming paradigm
* **Modular Design:** Easy updates and extensions
* **Performance Optimization:** Utilizes JAX's JIT compilation and vectorization
* **Validated Implementation:** >0.95 correlation with LigandMPNN reference across decoding paths

Validation
----------

Aminx has been rigorously validated against the upstream
`LigandMPNN <https://github.com/dauparas/LigandMPNN>`_ reference implementation:

.. list-table:: Equivalence Test Results
   :widths: 40 30 30
   :header-rows: 1

   * - Decoding Path
     - Correlation
     - Status
   * - Unconditional
     - 0.984
     - ✅ Validated
   * - Conditional
     - 0.958-0.984
     - ✅ Validated
   * - Autoregressive
     - 0.953-0.970
     - ✅ Validated

All three decoding paths achieve **>0.95 Pearson correlation** with reference outputs, ensuring
faithful reproduction of the original model's behavior.

Canonical parity/equivalence reports (aligned with README):

* `docs/FINAL_VALIDATION_RESULTS.md <https://github.com/maraxen/Aminx/blob/main/docs/FINAL_VALIDATION_RESULTS.md>`_
* `docs/parity/parity_report.md <https://github.com/maraxen/Aminx/blob/main/docs/parity/parity_report.md>`_

Rendered parity artifacts are also available as
`docs/parity/parity_report.html <https://github.com/maraxen/Aminx/blob/main/docs/parity/parity_report.html>`_
and
`docs/parity/parity_report.pdf <https://github.com/maraxen/Aminx/blob/main/docs/parity/parity_report.pdf>`_.

Installation
------------

.. code-block:: bash

  pip install -e .

For development:

.. code-block:: bash

  pip install -e ".[dev]"

Use notes
------------
