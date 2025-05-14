Mathematical Background
======================

Poincaré Ball Model
------------------

The Poincaré ball model is a model of hyperbolic geometry where the entire hyperbolic space is mapped to the interior of a Euclidean unit ball. 
The Poincaré ball has a negative curvature, which is represented by the parameter ``curvature`` in this library.

Key Operations
~~~~~~~~~~~~~

* **Möbius addition**: The equivalent of "adding" two points in hyperbolic space
* **Exponential map**: Mapping from the tangent space to the manifold
* **Logarithmic map**: Mapping from the manifold to the tangent space
* **Parallel transport**: Moving tangent vectors along geodesics

.. math::

   d(x, y) = \frac{2}{\sqrt{c}} \text{arctanh}(\sqrt{c} \| -x \oplus_c y \|)

Where :math:`\oplus_c` is the Möbius addition and :math:`c` is the curvature parameter.