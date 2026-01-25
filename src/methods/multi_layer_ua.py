"""
Multi-Layer Uncertainty-Aware (UA) CoT Vector implementation.

Extends the UA Vector method to support simultaneous injection
across multiple layers with independent Bayesian shrinkage gating.

Key features:
- Independent UA vectors for each target layer
- Layer-specific tau values for adaptive regularization
- Efficient batch extraction across layers
"""

import torch
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from .ua_vector import UACoTVector
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class MultiLayerUAVector(BaseCoTVectorMethod):
    """
    Multi-Layer UA CoT Vector with independent shrinkage per layer.
    
    Allows simultaneous vector injection at multiple layers,
    each with its own uncertainty-aware gating mechanism.
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_indices: List[int],
        dataset_type: str = "gsm8k",
        tau_squared: float = 1.0,
        layer_taus: Optional[List[float]] = None,
        min_variance: float = 1e-6,
    ):
        """
        Initialize Multi-Layer UA Vector extractor.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_indices: List of layer indices for extraction/injection
            dataset_type: Dataset type (gsm8k, math, mmlu_pro)
            tau_squared: Default prior variance τ² (used if layer_taus not provided)
            layer_taus: Optional per-layer tau values
            min_variance: Minimum variance for numerical stability
        """
        # Use first layer as primary for base class
        super().__init__(model_wrapper, tokenizer, layer_indices[0], dataset_type)
        
        self.layer_indices = sorted(layer_indices)
        self.num_layers = len(layer_indices)
        self.min_variance = min_variance
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Set per-layer tau values
        if layer_taus is not None:
            assert len(layer_taus) == len(layer_indices), \
                f"layer_taus length ({len(layer_taus)}) must match layer_indices ({len(layer_indices)})"
            self.layer_taus = layer_taus
        else:
            self.layer_taus = [tau_squared] * len(layer_indices)
        
        # Storage for layer-wise vectors and statistics
        self.layer_vectors: Dict[int, torch.Tensor] = {}
        self.layer_means: Dict[int, torch.Tensor] = {}
        self.layer_variances: Dict[int, torch.Tensor] = {}
        self.layer_shrinkage: Dict[int, torch.Tensor] = {}
    
    def extract_single_multi_layer(self, sample) -> Dict[int, torch.Tensor]:
        """
        Extract activation differences for all target layers from a single sample.
        
        Returns dict mapping layer_idx -> difference vector
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
        
        layer_diffs = {}
        
        # Extract activations for all target layers
        self.model_wrapper.clear_hooks()
        
        # Register extraction hooks for all layers
        for layer_idx in self.layer_indices:
            self.model_wrapper.register_extraction_hook(layer_idx)
        
        # Forward pass for CoT
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        
        # Collect CoT activations
        cot_activations = {}
        for layer_idx in self.layer_indices:
            act = self.model_wrapper.get_activations(layer_idx)
            # Move to correct device if needed
            cot_activations[layer_idx] = act[:, cot_answer_pos, :].mean(dim=1)
        
        # Clear and re-register for non-CoT
        self.model_wrapper.clear_hooks()
        for layer_idx in self.layer_indices:
            self.model_wrapper.register_extraction_hook(layer_idx)
        
        # Forward pass for non-CoT
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        
        # Compute differences for all layers
        for layer_idx in self.layer_indices:
            non_cot_act = self.model_wrapper.get_activations(layer_idx)
            non_cot_answer_act = non_cot_act[:, non_cot_answer_pos, :].mean(dim=1)
            
            diff = cot_activations[layer_idx] - non_cot_answer_act
            layer_diffs[layer_idx] = diff.squeeze(0)
        
        self.model_wrapper.clear_hooks()
        
        return layer_diffs
    
    def extract(self, support_samples: List) -> Dict[int, torch.Tensor]:
        """
        Extract UA CoT Vectors for all target layers.
        
        Args:
            support_samples: List of training samples
            
        Returns:
            Dictionary mapping layer_idx -> UA vector
        """
        print(f"Extracting Multi-Layer UA Vectors from {len(support_samples)} samples")
        print(f"  Target layers: {self.layer_indices}")
        print(f"  Per-layer τ²: {self.layer_taus}")
        
        # Collect vectors for each layer
        layer_vectors_list: Dict[int, List[torch.Tensor]] = {
            layer_idx: [] for layer_idx in self.layer_indices
        }
        
        for sample in tqdm(support_samples, desc="Extracting", ncols=100):
            try:
                layer_diffs = self.extract_single_multi_layer(sample)
                for layer_idx, diff in layer_diffs.items():
                    layer_vectors_list[layer_idx].append(diff)
            except Exception as e:
                continue
        
        # Process each layer
        for i, layer_idx in enumerate(self.layer_indices):
            vectors = layer_vectors_list[layer_idx]
            
            if not vectors:
                print(f"  Warning: No vectors extracted for layer {layer_idx}")
                continue
            
            # Stack vectors for this layer
            stacked = torch.stack(vectors, dim=0)  # [N, hidden]
            
            # Compute mean
            mean_vec = stacked.mean(dim=0)
            self.layer_means[layer_idx] = mean_vec
            
            # Compute variance
            if len(vectors) > 1:
                var_vec = stacked.var(dim=0, unbiased=True)
            else:
                var_vec = torch.full_like(mean_vec, self.min_variance)
            var_vec = torch.clamp(var_vec, min=self.min_variance)
            self.layer_variances[layer_idx] = var_vec
            
            # Compute shrinkage coefficients
            tau_sq = self.layer_taus[i]
            k = var_vec / (var_vec + tau_sq)
            self.layer_shrinkage[layer_idx] = k
            
            # Apply shrinkage
            ua_vec = k * mean_vec
            self.layer_vectors[layer_idx] = ua_vec
            
            # Print layer statistics
            print(f"\n  Layer {layer_idx}:")
            print(f"    Mean norm: {mean_vec.norm().item():.4f}")
            print(f"    UA norm: {ua_vec.norm().item():.4f}")
            print(f"    Shrinkage ratio: {(ua_vec.norm() / mean_vec.norm()).item():.4f}")
            print(f"    k stats: mean={k.mean().item():.3f}, min={k.min().item():.3f}, max={k.max().item():.3f}")
        
        # Set primary vector (first layer) for compatibility
        self.vector = self.layer_vectors.get(self.layer_indices[0])
        
        return self.layer_vectors
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the primary (first layer) UA vector."""
        return self.vector
    
    def get_layer_vector(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get UA vector for a specific layer."""
        return self.layer_vectors.get(layer_idx)
    
    def get_all_vectors(self) -> Dict[int, torch.Tensor]:
        """Get all layer vectors."""
        return self.layer_vectors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics for all layers."""
        return {
            "layer_indices": self.layer_indices,
            "layer_taus": self.layer_taus,
            "layer_means": {k: v.cpu() for k, v in self.layer_means.items()},
            "layer_variances": {k: v.cpu() for k, v in self.layer_variances.items()},
            "layer_shrinkage": {k: v.cpu() for k, v in self.layer_shrinkage.items()},
            "layer_vectors": {k: v.cpu() for k, v in self.layer_vectors.items()},
        }


class MultiLayerEvaluator:
    """
    Evaluator for multi-layer UA vector injection.
    
    Supports simultaneous injection at multiple layers
    with configurable per-layer scaling factors.
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_vectors: Dict[int, torch.Tensor],
        layer_weights: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize multi-layer evaluator.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_vectors: Dict mapping layer_idx -> vector
            layer_weights: Optional per-layer scaling factors
        """
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.layer_vectors = layer_vectors
        
        # Default weights = 1.0 for all layers
        if layer_weights is None:
            self.layer_weights = {k: 1.0 for k in layer_vectors.keys()}
        else:
            self.layer_weights = layer_weights
    
    def register_all_hooks(self):
        """Register injection hooks for all layers."""
        self.model_wrapper.clear_hooks()
        
        for layer_idx, vector in self.layer_vectors.items():
            weight = self.layer_weights.get(layer_idx, 1.0)
            self.model_wrapper.register_injection_hook(
                layer_idx=layer_idx,
                vector=vector,
                scaling_factor=weight,
                requires_grad=False
            )
    
    def clear_hooks(self):
        """Clear all injection hooks."""
        self.model_wrapper.clear_hooks()
