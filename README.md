# HypTorch

[![Documentation Status](https://readthedocs.org/projects/hyptorch/badge/?version=latest)](https://hyptorch.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/hyptorch.svg)](https://badge.fury.io/py/hyptorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A PyTorch library for hyperbolic deep learning. HypTorch provides hyperbolic neural network layers, optimizers, and utilities for building deep learning models in hyperbolic space, with a focus on the Poincaré ball model.

## 🌟 Key Features

- **Hyperbolic Neural Layers**: Drop-in replacements for standard PyTorch layers

  - `HypLinear`: Hyperbolic linear transformation using Möbius operations
  - `HyperbolicMLR`: Multi-class logistic regression in hyperbolic space
  - `ConcatPoincareLayer`: Hyperbolic concatenation layer
  - `HyperbolicDistanceLayer`: Compute geodesic distances

- **Manifold Operations**: Full suite of hyperbolic geometry operations

  - Exponential and logarithmic maps
  - Möbius addition and matrix-vector multiplication
  - Geodesic distances and Riemannian metrics
  - Projections and embeddings

- **Seamless PyTorch Integration**:

  - Compatible with PyTorch's autograd system
  - Support for standard optimizers with Riemannian gradients
  - Easy transitions between Euclidean and hyperbolic spaces

- **Numerical Stability**: Careful handling of the boundary conditions and numerical precision issues inherent to hyperbolic geometry

## 📦 Installation

### From PyPI

```bash
pip install hyptorch
```

### From Source

```bash
git clone https://github.com/Iarrova/hyptorch.git
cd hyptorch
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev,docs]"
```

## 🚀 Quick Start

```python
import torch
from hyptorch import PoincareBall, HypLinear, HyperbolicMLR

# Create a Poincaré ball manifold
manifold = PoincareBall(curvature=1.0)

# Create hyperbolic layers
hyp_linear = HypLinear(in_features=10, out_features=5, manifold=manifold)
hyp_mlr = HyperbolicMLR(ball_dim=5, n_classes=3, manifold=manifold)

# Forward pass
x = torch.randn(32, 10) * 0.1  # Keep inputs small for numerical stability
h = hyp_linear(x)  # Hyperbolic linear transformation
logits = hyp_mlr(h)  # Hyperbolic MLR
probs = torch.softmax(logits, dim=1)  # Standard softmax works!
```

## 📖 Detailed Examples

### Building a Hyperbolic Neural Network

```python
import torch
import torch.nn as nn
from hyptorch import PoincareBall, HypLinear, ToPoincare, FromPoincare

class HyperbolicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, curvature=1.0):
        super().__init__()
        self.manifold = PoincareBall(curvature=curvature)

        # Map from Euclidean to hyperbolic space
        self.to_hyperbolic = ToPoincare(self.manifold)

        # Hyperbolic layers
        self.hyp_layers = nn.Sequential(
            HypLinear(input_dim, hidden_dim, manifold=self.manifold),
            HypLinear(hidden_dim, hidden_dim, manifold=self.manifold),
            HypLinear(hidden_dim, output_dim, manifold=self.manifold)
        )

        # Map back to Euclidean for standard loss functions
        self.from_hyperbolic = FromPoincare(self.manifold)

    def forward(self, x):
        x = self.to_hyperbolic(x)
        x = self.hyp_layers(x)
        x = self.from_hyperbolic(x)
        return x

# Create and use the model
model = HyperbolicNet(input_dim=20, hidden_dim=10, output_dim=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
x = torch.randn(32, 20)
y = torch.randint(0, 5, (32,))
criterion = nn.CrossEntropyLoss()

output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

### Working with Hyperbolic Embeddings

```python
from hyptorch import PoincareBall, HyperbolicDistanceLayer
from hyptorch.operations import HyperbolicMean

# Create manifold and embeddings
manifold = PoincareBall(curvature=1.0)
embeddings = torch.randn(100, 10) * 0.1  # 100 embeddings in 10D
embeddings = manifold.project(embeddings)

# Compute pairwise distances
dist_layer = HyperbolicDistanceLayer(manifold)
distances = dist_layer(embeddings[0:10], embeddings[10:20])

# Compute hyperbolic mean
mean_op = HyperbolicMean(manifold)
cluster_center = mean_op(embeddings[0:10])

# Find nearest neighbors using hyperbolic distance
query = embeddings[0]
all_distances = torch.stack([manifold.distance(query, emb) for emb in embeddings])
nearest_indices = torch.argsort(all_distances)[:5]
```

### Hyperbolic Classification

```python
from hyptorch import HyperbolicMLR, ToPoincare

# Hyperbolic multi-class classifier
class HyperbolicClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, curvature=1.0):
        super().__init__()
        self.manifold = PoincareBall(curvature=curvature)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.to_hyperbolic = ToPoincare(self.manifold)
        self.classifier = HyperbolicMLR(
            ball_dim=16,
            n_classes=num_classes,
            manifold=self.manifold
        )

    def forward(self, x):
        # Encode in Euclidean space
        features = self.encoder(x)
        # Map to hyperbolic space
        hyp_features = self.to_hyperbolic(features)
        # Classify in hyperbolic space
        logits = self.classifier(hyp_features)
        return logits

# Usage
model = HyperbolicClassifier(input_dim=784, num_classes=10)
x = torch.randn(32, 784)
logits = model(x)
predictions = torch.argmax(logits, dim=1)
```

## 🔬 Mathematical Background

HypTorch implements neural networks in hyperbolic space, specifically the Poincaré ball model. Key concepts:

- **Poincaré Ball**: A model of hyperbolic geometry where the space is contained within a unit ball
- **Möbius Operations**: Generalizations of vector addition and matrix multiplication that respect hyperbolic geometry
- **Riemannian Metrics**: The library handles the non-Euclidean metric tensor, ensuring proper gradient flow
- **Geodesics**: Shortest paths in hyperbolic space, analogous to straight lines in Euclidean space

For more details, see our [documentation](https://hyptorch.readthedocs.io/en/latest/).

## 📚 API Overview

### Manifolds

- `PoincareBall`: The Poincaré ball model of hyperbolic space

### Neural Network Layers

- `HypLinear`: Hyperbolic linear layer
- `HyperbolicMLR`: Hyperbolic multinomial logistic regression
- `ConcatPoincareLayer`: Concatenation in hyperbolic space
- `HyperbolicDistanceLayer`: Geodesic distance computation
- `ToPoincare`: Map from Euclidean to hyperbolic space
- `FromPoincare`: Map from hyperbolic to Euclidean space

### Operations

- `HyperbolicMean`: Fréchet mean in hyperbolic space
- Tensor operations: `norm`, `squared_norm`, `dot_product`
- Transformations: `PoincareToKleinTransform`, `KleinToPoincareTransform`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This library builds upon the theoretical foundations laid by:

- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112) (Ganea et al., 2018)
- [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039) (Nickel & Kiela, 2017)
- [Hyperbolic Image Embeddings](https://arxiv.org/abs/1904.02239) (Khrulkov et al., 2020)

## 📞 Contact

- **Author**: Ian Roberts Valenzuela
- **Documentation**: [https://hyptorch.readthedocs.io](https://hyptorch.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Iarrova/hyptorch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Iarrova/hyptorch/discussions)
