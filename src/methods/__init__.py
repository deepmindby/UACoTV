"""
CoT Vector methods package.

Available methods based on Variational CoT Vectors framework:

- ExtractedCoTVector: Extract vectors from activation differences (Eq. 4-5)
  Statistical aggregation to approximate the posterior distribution.

- LearnableCoTVector: Learn vectors via teacher-student framework (Eq. 6)
  Gradient optimization for learning global reasoning patterns.

- UACoTVector: Uncertainty-Aware vectors with Bayesian shrinkage
  MAP estimation with structured prior for adaptive gating.

- OptimizationUACoTVector: Optimization-based UA vectors
  Uses gradient descent with precision-weighted loss for implicit regularization.
  Avoids hard thresholding by using early stopping as implicit regularization.

- MixtureUACoTVector: Mixture of Uncertainty-Aware vectors using IGMM
  Dirichlet Process GMM for discovering multimodal reasoning patterns,
  with per-cluster Bayesian shrinkage.

- MultiLayerUAVector: Multi-layer UA injection
  Simultaneous injection across multiple layers with independent shrinkage.

Note: RL-based methods (GRPO, DAPO, Self-Evolved) have been removed.
The focus is now on analytical/mathematical approaches.
"""

from .base import BaseCoTVectorMethod
from .extracted import ExtractedCoTVector
from .learnable import LearnableCoTVector
from .ua_vector import UACoTVector
from .optimization_ua import OptimizationUACoTVector
from .mixture_ua import MixtureUACoTVector
from .multi_layer_ua import MultiLayerUAVector, MultiLayerEvaluator

__all__ = [
    "BaseCoTVectorMethod",
    "ExtractedCoTVector",
    "LearnableCoTVector",
    "UACoTVector",
    "OptimizationUACoTVector",
    "MixtureUACoTVector",
    "MultiLayerUAVector",
    "MultiLayerEvaluator",
]