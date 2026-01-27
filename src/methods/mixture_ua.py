"""
Mixture of Uncertainty-Aware (MixtureUA) CoT Vector implementation.

Based on Infinite Gaussian Mixture Model (IGMM) with Dirichlet Process prior.

Key insight: The reasoning space is multimodal - different reasoning paths
exist for different types of problems. This method:
1. Discovers multiple reasoning modes via DPGMM clustering
2. Applies per-cluster Bayesian shrinkage
3. Returns the dominant mode while preserving all discovered modes

Formula for per-cluster shrinkage:
    λ_k = τ² / (τ² + σ_k²)
    v_k* = λ_k ⊙ μ_k

Where:
- Low variance σ² → λ ≈ 1 → dimension preserved
- High variance σ² → λ ≈ 0 → dimension suppressed
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class MixtureUACoTVector(BaseCoTVectorMethod):
    """
    Mixture of Uncertainty-Aware CoT Vectors using DPGMM clustering.
    
    Unlike single-Gaussian UACoTVector, this method assumes the reasoning
    space is multimodal and uses a Dirichlet Process prior to automatically
    discover multiple reasoning modes.
    
    Each discovered mode gets its own uncertainty-aware shrinkage treatment,
    allowing the method to capture diverse reasoning patterns while still
    suppressing noise within each mode.
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        tau_squared: float = 1.0,
        min_variance: float = 1e-6,
        num_components: int = 10,
        concentration_prior: float = 1.0,
        min_cluster_weight: float = 0.0001,
    ):
        """
        Initialize Mixture UA CoT Vector extractor.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_idx: Layer index for extraction
            dataset_type: Dataset type (gsm8k, math, mmlu_pro)
            tau_squared: Prior variance τ² for Bayesian shrinkage
            min_variance: Minimum variance for numerical stability
            num_components: Upper bound for DPGMM components
            concentration_prior: Concentration prior α for Dirichlet Process
                                 (smaller = fewer clusters, larger = more clusters)
            min_cluster_weight: Minimum weight threshold for valid clusters
        """
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.tau_squared = tau_squared
        self.min_variance = min_variance
        self.num_components = num_components
        self.concentration_prior = concentration_prior
        self.min_cluster_weight = min_cluster_weight
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Storage for results
        self.raw_vectors: Optional[torch.Tensor] = None
        self.cluster_vectors: Dict[int, torch.Tensor] = {}  # Per-cluster shrunk vectors
        self.cluster_means: Dict[int, torch.Tensor] = {}
        self.cluster_variances: Dict[int, torch.Tensor] = {}
        self.cluster_shrinkage: Dict[int, torch.Tensor] = {}
        self.cluster_weights: Dict[int, float] = {}
        self.cluster_labels: Optional[np.ndarray] = None
        self.dominant_cluster_id: Optional[int] = None
        self.dpgmm_model = None
    
    def extract_single(self, sample) -> torch.Tensor:
        """
        Extract activation difference for a single sample.
        Same as UACoTVector.extract_single().
        """
        device = self.model_wrapper.device
        
        # Build prompts
        if self.dataset_type == "mmlu_pro":
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        # Tokenize
        cot_encoding = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        non_cot_encoding = self.tokenizer(non_cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        cot_ids = cot_encoding["input_ids"].to(device)
        non_cot_ids = non_cot_encoding["input_ids"].to(device)
        cot_mask = cot_encoding["attention_mask"].to(device)
        non_cot_mask = non_cot_encoding["attention_mask"].to(device)
        
        # Find answer token positions
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        answer_len = answer_ids.shape[1]
        
        cot_answer_pos = list(range(cot_ids.shape[1] - answer_len, cot_ids.shape[1]))
        non_cot_answer_pos = list(range(non_cot_ids.shape[1] - answer_len, non_cot_ids.shape[1]))
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        # Extract CoT activations
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        cot_answer_activation = cot_activation[:, cot_answer_pos, :].mean(dim=1)
        
        # Clear and extract non-CoT activations
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        non_cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        non_cot_answer_activation = non_cot_activation[:, non_cot_answer_pos, :].mean(dim=1)
        
        # Compute difference
        diff = cot_answer_activation - non_cot_answer_activation
        
        self.model_wrapper.clear_hooks()
        
        return diff.squeeze(0)  # [hidden]
    
    def _fit_dpgmm(self, vectors_np: np.ndarray) -> None:
        """
        Fit Dirichlet Process Gaussian Mixture Model to the vectors.
        
        Args:
            vectors_np: Numpy array of shape [N, hidden_dim]
        """
        from sklearn.mixture import BayesianGaussianMixture
        
        print(f"\nFitting DPGMM with up to {self.num_components} components...")
        print(f"  Concentration prior α = {self.concentration_prior}")
        print(f"  Input shape: {vectors_np.shape}")
        
        # Initialize and fit DPGMM
        self.dpgmm_model = BayesianGaussianMixture(
            n_components=self.num_components,
            covariance_type='diag',  # Use diagonal covariance for efficiency
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=self.concentration_prior,
            max_iter=200,
            n_init=3,
            random_state=42,
            verbose=0,
        )
        
        self.dpgmm_model.fit(vectors_np)
        self.cluster_labels = self.dpgmm_model.predict(vectors_np)
        
        # Get cluster weights
        weights = self.dpgmm_model.weights_
        
        # Count effective clusters (weight > threshold)
        effective_clusters = np.sum(weights > self.min_cluster_weight)
        print(f"  Effective clusters (weight > {self.min_cluster_weight}): {effective_clusters}")
        
        # Print weight distribution
        sorted_weights = np.sort(weights)[::-1]
        print(f"  Top-5 cluster weights: {sorted_weights[:5]}")
    
    def _compute_per_cluster_shrinkage(self, vectors: torch.Tensor) -> None:
        """
        Compute Bayesian shrinkage for each valid cluster.
        
        For each cluster k:
        1. Compute mean μ_k from samples in cluster k
        2. Compute variance σ_k² from samples in cluster k
        3. Apply shrinkage: λ_k = τ² / (τ² + σ_k²)
        4. Compute shrunk vector: v_k* = λ_k ⊙ μ_k
        
        Args:
            vectors: Tensor of shape [N, hidden_dim]
        """
        weights = self.dpgmm_model.weights_
        unique_labels = np.unique(self.cluster_labels)
        
        print(f"\nComputing per-cluster shrinkage (τ² = {self.tau_squared})...")
        
        for cluster_id in unique_labels:
            # Get weight for this cluster
            weight = weights[cluster_id]
            
            # Skip clusters with insufficient weight
            if weight < self.min_cluster_weight:
                continue
            
            # Get samples belonging to this cluster
            mask = self.cluster_labels == cluster_id
            cluster_samples = vectors[mask]  # [n_k, hidden]
            n_samples = cluster_samples.shape[0]
            
            if n_samples == 0:
                continue
            
            self.cluster_weights[cluster_id] = float(weight)
            
            # Compute cluster mean
            mean_vec = cluster_samples.mean(dim=0)
            self.cluster_means[cluster_id] = mean_vec
            
            # Compute cluster variance
            if n_samples > 1:
                var_vec = cluster_samples.var(dim=0, unbiased=True)
            else:
                var_vec = torch.full_like(mean_vec, self.min_variance)
            var_vec = torch.clamp(var_vec, min=self.min_variance)
            self.cluster_variances[cluster_id] = var_vec
            
            # Compute shrinkage coefficient: λ = τ² / (τ² + σ²)
            # When σ² is small (stable signal): λ → 1 (preserve)
            # When σ² is large (noise): λ → 0 (suppress)
            shrinkage_coef = self.tau_squared / (self.tau_squared + var_vec)
            self.cluster_shrinkage[cluster_id] = shrinkage_coef
            
            # Apply shrinkage: v* = λ ⊙ μ
            shrunk_vec = shrinkage_coef * mean_vec
            self.cluster_vectors[cluster_id] = shrunk_vec
            
            # Print cluster statistics
            print(f"\n  Cluster {cluster_id} (weight={weight:.4f}, n={n_samples}):")
            print(f"    Mean norm: {mean_vec.norm().item():.4f}")
            print(f"    Shrunk norm: {shrunk_vec.norm().item():.4f}")
            print(f"    Shrinkage ratio: {(shrunk_vec.norm() / (mean_vec.norm() + 1e-8)).item():.4f}")
            print(f"    λ stats: mean={shrinkage_coef.mean().item():.3f}, "
                  f"min={shrinkage_coef.min().item():.3f}, "
                  f"max={shrinkage_coef.max().item():.3f}")
    
    def _select_dominant_mode(self) -> torch.Tensor:
        """
        Select the dominant mode (cluster with highest weight).
        
        Returns:
            The shrunk vector of the dominant cluster
        """
        if not self.cluster_weights:
            raise ValueError("No valid clusters found!")
        
        # Find cluster with maximum weight
        self.dominant_cluster_id = max(self.cluster_weights.keys(), 
                                        key=lambda k: self.cluster_weights[k])
        
        dominant_weight = self.cluster_weights[self.dominant_cluster_id]
        dominant_vector = self.cluster_vectors[self.dominant_cluster_id]
        
        print(f"\n★ Dominant cluster: {self.dominant_cluster_id} "
              f"(weight={dominant_weight:.4f}, norm={dominant_vector.norm().item():.4f})")
        
        return dominant_vector
    
    def extract(self, support_samples: List) -> torch.Tensor:
        """
        Extract Mixture UA CoT Vector from support set.
        
        Steps:
        1. Extract raw difference vectors for all samples
        2. Cluster using DPGMM
        3. Apply per-cluster Bayesian shrinkage
        4. Return dominant mode vector
        
        Args:
            support_samples: List of training samples
            
        Returns:
            Dominant mode vector (the primary result)
            
        Note:
            All cluster vectors are stored in self.cluster_vectors
            for later saving/analysis.
        """
        print(f"Extracting Mixture UA CoT Vectors from {len(support_samples)} samples at layer {self.layer_idx}...")
        print(f"  DPGMM components (max): {self.num_components}")
        print(f"  Concentration prior α: {self.concentration_prior}")
        print(f"  Prior variance τ²: {self.tau_squared}")
        
        # Step 1: Extract raw difference vectors
        vectors = []
        for sample in tqdm(support_samples, desc="Extracting raw vectors", ncols=100):
            try:
                vec = self.extract_single(sample)
                vectors.append(vec)
            except Exception as e:
                continue
        
        if not vectors:
            raise ValueError("No vectors extracted!")
        
        # Stack all vectors
        self.raw_vectors = torch.stack(vectors, dim=0)  # [N, hidden]
        print(f"\nExtracted {self.raw_vectors.shape[0]} raw vectors")
        
        # Step 2: Fit DPGMM clustering
        vectors_np = self.raw_vectors.cpu().numpy().astype(np.float64)
        self._fit_dpgmm(vectors_np)
        
        # Step 3: Compute per-cluster shrinkage
        self._compute_per_cluster_shrinkage(self.raw_vectors)
        
        # Step 4: Select dominant mode
        self.vector = self._select_dominant_mode()
        
        # Print summary
        self._print_summary()
        
        return self.vector
    
    def _print_summary(self):
        """Print extraction summary."""
        print("\n" + "=" * 60)
        print("Mixture UA Extraction Summary")
        print("=" * 60)
        print(f"Total samples: {self.raw_vectors.shape[0]}")
        print(f"Valid clusters: {len(self.cluster_vectors)}")
        print(f"Dominant cluster: {self.dominant_cluster_id}")
        
        print("\nCluster breakdown:")
        for cid in sorted(self.cluster_vectors.keys(), 
                         key=lambda k: self.cluster_weights[k], 
                         reverse=True):
            weight = self.cluster_weights[cid]
            vec = self.cluster_vectors[cid]
            n_samples = np.sum(self.cluster_labels == cid)
            marker = "★" if cid == self.dominant_cluster_id else " "
            print(f"  {marker} Cluster {cid}: weight={weight:.4f}, n={n_samples}, "
                  f"norm={vec.norm().item():.4f}")
        print("=" * 60)
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the dominant cluster vector."""
        return self.vector
    
    def get_cluster_vector(self, cluster_id: int) -> Optional[torch.Tensor]:
        """Get vector for a specific cluster."""
        return self.cluster_vectors.get(cluster_id)
    
    def get_all_cluster_vectors(self) -> Dict[int, torch.Tensor]:
        """Get all cluster vectors."""
        return self.cluster_vectors
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics for analysis.
        
        Returns:
            Dictionary containing all extraction results
        """
        stats = {
            "layer_idx": self.layer_idx,
            "tau_squared": self.tau_squared,
            "num_components": self.num_components,
            "concentration_prior": self.concentration_prior,
            "min_cluster_weight": self.min_cluster_weight,
            "dominant_cluster_id": self.dominant_cluster_id,
            "num_valid_clusters": len(self.cluster_vectors),
        }
        
        if self.raw_vectors is not None:
            stats["num_samples"] = self.raw_vectors.shape[0]
            stats["raw_vectors"] = self.raw_vectors.cpu()
        
        if self.cluster_labels is not None:
            stats["cluster_labels"] = self.cluster_labels
        
        # Per-cluster statistics
        stats["cluster_weights"] = self.cluster_weights.copy()
        stats["cluster_vectors"] = {k: v.cpu() for k, v in self.cluster_vectors.items()}
        stats["cluster_means"] = {k: v.cpu() for k, v in self.cluster_means.items()}
        stats["cluster_variances"] = {k: v.cpu() for k, v in self.cluster_variances.items()}
        stats["cluster_shrinkage"] = {k: v.cpu() for k, v in self.cluster_shrinkage.items()}
        
        return stats