"""
Optimization-based Uncertainty-Aware (UA) CoT Vector implementation.

Based on "Implicit Regularization" theory from:
- "Whoever Started the Interference Should End It"
- "UA CoT Vectors - Algorithm 2"

Key insight: Instead of using analytic Bayesian shrinkage (hard thresholding),
this method uses gradient-based optimization with early stopping to implicitly
filter out noise. High variance dimensions receive low gradient weights and
stay close to zero (suppressed), while low variance dimensions quickly converge
to the mean (preserved).

Algorithm:
1. Initialize v* = 0 (not mean!)
2. Compute precision weights W = 1 / (σ² + ε)
3. Optimize weighted MSE: L = 0.5 * Σ(W ⊙ (v* - μ)²)
4. Early stopping via limited iterations acts as implicit regularization
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class OptimizationUACoTVector(BaseCoTVectorMethod):
    """
    Optimization-based Uncertainty-Aware CoT Vector.
    
    Uses gradient descent with precision-weighted loss to implicitly
    perform regularization. The key difference from analytic UA:
    - Analytic UA: z = k ⊙ μ where k = τ² / (σ² + τ²) [hard thresholding]
    - Optimization UA: z = argmin_v 0.5 * Σ(W ⊙ (v - μ)²) with early stopping
    
    The optimization naturally suppresses high-variance (noisy) dimensions
    because they have low weights and thus small gradients.
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        learning_rate: float = 5e-3,
        num_steps: int = 5,
        min_variance: float = 1e-6,
    ):
        """
        Initialize Optimization-based UA CoT Vector extractor.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_idx: Layer index for extraction
            dataset_type: Dataset type (gsm8k, math, mmlu_pro)
            learning_rate: Learning rate η for Adam optimizer
            num_steps: Number of optimization steps T (using num_epochs arg)
            min_variance: Minimum variance ε for numerical stability
        """
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.min_variance = min_variance
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Store statistics for analysis
        self.mean_vector: Optional[torch.Tensor] = None
        self.variance_vector: Optional[torch.Tensor] = None
        self.precision_weights: Optional[torch.Tensor] = None
        self.optimization_history: List[float] = []
    
    def extract_single(self, sample) -> torch.Tensor:
        """
        Extract activation difference for a single sample.
        
        Returns the difference: α_CoT(answer) - α_NonCoT(answer)
        This represents the "teacher hidden state" for this sample.
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
        
        # Find answer token positions (last N tokens where N = answer length)
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        answer_len = answer_ids.shape[1]
        
        # Get positions for answer tokens
        cot_answer_pos = list(range(cot_ids.shape[1] - answer_len, cot_ids.shape[1]))
        non_cot_answer_pos = list(range(non_cot_ids.shape[1] - answer_len, non_cot_ids.shape[1]))
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        # Extract CoT activations
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        cot_answer_activation = cot_activation[:, cot_answer_pos, :].mean(dim=1)  # [1, hidden]
        
        # Clear and extract non-CoT activations
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        non_cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        non_cot_answer_activation = non_cot_activation[:, non_cot_answer_pos, :].mean(dim=1)  # [1, hidden]
        
        # Compute difference (teacher hidden state for this sample)
        diff = cot_answer_activation - non_cot_answer_activation  # [1, hidden]
        
        self.model_wrapper.clear_hooks()
        
        return diff.squeeze(0)  # [hidden]
    
    def extract(self, support_samples: List) -> torch.Tensor:
        """
        Extract Optimization-based UA CoT Vector from support set.
        
        Algorithm (based on Algorithm 2):
        1. Collect raw difference vectors v^(i) for all support samples
        2. Compute support mean μ_D (target) and variance σ²_D (uncertainty)
        3. Compute precision weights W = 1 / (σ² + ε)
        4. Initialize v* = 0 (zeros, NOT mean!)
        5. Optimize: minimize L = 0.5 * Σ(W ⊙ (v* - μ)²) for T steps
        6. Return optimized v*
        
        Args:
            support_samples: List of training samples
            
        Returns:
            Optimization-based UA CoT vector
        """
        print(f"Extracting Optimization-based UA CoT Vectors from {len(support_samples)} samples at layer {self.layer_idx}...")
        print(f"  Learning rate η = {self.learning_rate}")
        print(f"  Optimization steps T = {self.num_steps}")
        print(f"  Min variance ε = {self.min_variance}")
        
        # ========== Step 1: Data Collection ==========
        vectors = []
        for sample in tqdm(support_samples, desc="Extracting raw vectors", ncols=100):
            try:
                vec = self.extract_single(sample)
                vectors.append(vec)
            except Exception as e:
                # Skip problematic samples
                continue
        
        if not vectors:
            raise ValueError("No vectors extracted!")
        
        # Stack all sample vectors: [N, hidden]
        stacked = torch.stack(vectors, dim=0)
        device = stacked.device
        hidden_size = stacked.shape[1]
        
        print(f"  Collected {len(vectors)} vectors, hidden_size = {hidden_size}")
        
        # ========== Step 2: Compute Statistics ==========
        # Support mean μ_D (target)
        self.mean_vector = stacked.mean(dim=0)  # [hidden]
        
        # Support variance σ²_D (uncertainty)
        if len(vectors) > 1:
            self.variance_vector = stacked.var(dim=0, unbiased=True)  # [hidden]
        else:
            # Single sample: use minimum variance
            self.variance_vector = torch.full_like(self.mean_vector, self.min_variance)
        
        # Ensure minimum variance for numerical stability
        self.variance_vector = torch.clamp(self.variance_vector, min=self.min_variance)
        
        # ========== Step 3: Compute Precision Weights ==========
        # W = 1 / (σ² + ε)
        # High variance -> Low weight -> Small gradient -> Suppressed
        # Low variance -> High weight -> Large gradient -> Preserved
        self.precision_weights = 1.0 / (self.variance_vector + self.min_variance)
        
        # ========== Step 4: Optimization ==========
        # Initialize v* to ZEROS (critical: not mean!)
        # This is the key difference from analytic UA
        v_star = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=stacked.dtype))
        
        # Adam optimizer
        optimizer = torch.optim.Adam([v_star], lr=self.learning_rate)
        
        # Detach targets to avoid gradient issues
        target_mean = self.mean_vector.detach()
        weights = self.precision_weights.detach()
        
        print(f"\n  Starting optimization...")
        self.optimization_history = []
        
        # Optimization loop
        pbar = tqdm(range(self.num_steps), desc="Optimizing", ncols=100)
        for step in pbar:
            optimizer.zero_grad()
            
            # Weighted MSE Loss: L = 0.5 * Σ(W ⊙ (v* - μ)²)
            diff = v_star - target_mean
            weighted_squared_diff = weights * (diff ** 2)
            loss = 0.5 * weighted_squared_diff.sum()
            
            # Backpropagate
            loss.backward()
            
            # Update
            optimizer.step()
            
            # Track history
            loss_val = loss.item()
            self.optimization_history.append(loss_val)
            
            # Update progress bar
            v_norm = v_star.detach().norm().item()
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "norm": f"{v_norm:.4f}"})
        
        # ========== Step 5: Return ==========
        self.vector = v_star.detach().clone()
        
        # Print statistics
        self._print_statistics()
        
        return self.vector
    
    def _print_statistics(self):
        """Print diagnostic statistics about the extracted vector."""
        print(f"\nOptimization UA Vector Statistics:")
        print(f"  Mean vector norm: {self.mean_vector.norm().item():.4f}")
        print(f"  Optimized vector norm: {self.vector.norm().item():.4f}")
        print(f"  Shrinkage ratio: {(self.vector.norm() / (self.mean_vector.norm() + 1e-8)).item():.4f}")
        
        # Precision weights statistics
        W = self.precision_weights
        print(f"\nPrecision Weights (W = 1/σ²):")
        print(f"  Mean: {W.mean().item():.4f}")
        print(f"  Std: {W.std().item():.4f}")
        print(f"  Min: {W.min().item():.4f}")
        print(f"  Max: {W.max().item():.4f}")
        
        # Variance statistics
        var = self.variance_vector
        print(f"\nVariance Statistics (σ²):")
        print(f"  Mean: {var.mean().item():.6f}")
        print(f"  Std: {var.std().item():.6f}")
        print(f"  Min: {var.min().item():.6f}")
        print(f"  Max: {var.max().item():.6f}")
        
        # Convergence analysis
        if self.optimization_history:
            print(f"\nOptimization Convergence:")
            print(f"  Initial loss: {self.optimization_history[0]:.4f}")
            print(f"  Final loss: {self.optimization_history[-1]:.4f}")
            print(f"  Reduction: {(1 - self.optimization_history[-1] / (self.optimization_history[0] + 1e-8)) * 100:.2f}%")
        
        # Per-dimension analysis: compare optimized vs mean
        relative_magnitude = torch.abs(self.vector) / (torch.abs(self.mean_vector) + 1e-8)
        highly_suppressed = (relative_magnitude < 0.1).sum().item()
        moderately_kept = ((relative_magnitude >= 0.1) & (relative_magnitude < 0.5)).sum().item()
        well_preserved = (relative_magnitude >= 0.5).sum().item()
        
        total_dims = relative_magnitude.numel()
        print(f"\nDimension Classification (|v*| / |μ|):")
        print(f"  Highly suppressed (< 0.1): {highly_suppressed} ({100*highly_suppressed/total_dims:.1f}%)")
        print(f"  Moderately kept (0.1 - 0.5): {moderately_kept} ({100*moderately_kept/total_dims:.1f}%)")
        print(f"  Well preserved (≥ 0.5): {well_preserved} ({100*well_preserved/total_dims:.1f}%)")
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the extracted Optimization UA CoT vector."""
        return self.vector
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics for analysis.
        
        Returns:
            Dictionary containing mean, variance, weights, and optimization history
        """
        if self.mean_vector is None:
            return {}
        
        return {
            "mean_vector": self.mean_vector.cpu(),
            "variance_vector": self.variance_vector.cpu(),
            "precision_weights": self.precision_weights.cpu(),
            "optimization_history": self.optimization_history,
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
            "min_variance": self.min_variance,
            "optimized_vector": self.vector.cpu() if self.vector is not None else None,
        }