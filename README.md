A suite of hyperbolic neural networks in PyTorch, primarily focused on the Poincaré ball model of hyperbolic geometry.

# Overview
HypTorch provides tools for working with neural networks in hyperbolic space, including:

- Poincaré ball model operations
- Hyperbolic layers and modules
- Distance calculations in hyperbolic space
- Mappings between models (Poincaré to Klein and vice versa)
- Manifold abstractions for geometric operations

# Installation
```bash
pip install hyptorch
```

# Mathematical Background

## Poincaré Ball Model
The Poincaré ball model is a model of hyperbolic geometry where the entire hyperbolic space is mapped to the interior of a Euclidean unit ball. The Poincaré ball has a negative curvature, which is represented by the parameter curvature in this library.

Some key operations include:
- Möbius addition: The equivalent of "adding" two points in hyperbolic space
- Exponential map: Mapping from the tangent space to the manifold
- Logarithmic map: Mapping from the manifold to the tangent space
- Parallel transport: Moving tangent vectors along geodesics

## Applications
Hyperbolic neural networks are particularly effective for:

- Hierarchical data structures
- Tree-like data
- Network/graph embedding
- Natural language processing
- Any data with inherent hierarchical structure