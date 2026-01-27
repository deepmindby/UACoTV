"""
Main entry point for CoT Vectors.

Supports four methods based on Variational CoT Vectors framework:
- Extracted: Statistical aggregation of activation differences
- Learnable: Gradient optimization via teacher-student framework
- UA: Uncertainty-Aware with Bayesian shrinkage
- Mixture UA: IGMM-based multimodal reasoning with per-cluster shrinkage

RL-based methods have been removed in favor of analytical approaches.
"""

import os
import torch
from datetime import datetime

from src.args import parse_args
from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.methods.extracted import ExtractedCoTVector
from src.methods.learnable import LearnableCoTVector
from src.methods.ua_vector import UACoTVector
from src.methods.mixture_ua import MixtureUACoTVector
from src.methods.multi_layer_ua import MultiLayerUAVector, MultiLayerEvaluator
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import set_seed, setup_wandb


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("CoT Vectors: Variational Framework")
    print("=" * 60)
    print(f"Model: {args.model_path.split('/')[-1]}")
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Layer: {args.layer_idx}")
    print(f"Mode: {args.mode}")
    print(f"Beams: {args.num_beams}, Max tokens: {args.max_new_tokens}")
    
    # Print method-specific config
    if args.method == "learnable":
        print("-" * 60)
        print("Learnable Configuration:")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Lambda: {args.lambda_val}")
        print(f"  Max length: {args.max_length}")
    
    if args.method == "ua":
        print("-" * 60)
        print("Uncertainty-Aware (UA) Configuration:")
        print(f"  Prior variance τ²: {args.tau_squared}")
        print(f"  Min variance: {args.min_variance}")
        if args.multi_layer:
            print(f"  Multi-layer: {args.target_layers}")
    
    if args.method == "mixture_ua":
        print("-" * 60)
        print("Mixture UA (IGMM) Configuration:")
        print(f"  Prior variance τ²: {args.tau_squared}")
        print(f"  Min variance: {args.min_variance}")
        print(f"  Max components: {args.num_mixture_components}")
        print(f"  Concentration prior α: {args.mixture_concentration}")
    
    print("=" * 60)
    
    # Setup WandB
    wandb_run = None
    if args.use_wandb:
        wandb_run = setup_wandb(args)
    
    # Load model
    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    print(f"Model loaded: {model_wrapper.num_layers} layers, hidden_size={model_wrapper.hidden_size}")
    
    # Load data
    print("\nLoading data...")
    support_samples = None
    test_samples = None
    
    if args.mode in ["extract", "train", "both"]:
        support_samples = load_dataset(
            args.data_path, args.dataset, "train", args.num_support_samples
        )
        print(f"Support set: {len(support_samples)} samples")
    
    if args.mode in ["eval", "both"]:
        test_samples = load_dataset(
            args.data_path, args.dataset, "test", args.num_test_samples
        )
        print(f"Test set: {len(test_samples)} samples")
    
    # Get or load vector
    vector = None
    layer_vectors = None  # For multi-layer
    cluster_vectors = None  # For mixture_ua
    method = None  # Keep reference to method object
    
    if args.vector_path:
        print(f"\nLoading vector from {args.vector_path}")
        loaded = torch.load(args.vector_path, map_location="cpu")
        if isinstance(loaded, dict):
            if "vector" in loaded:
                vector = loaded["vector"]
            elif "layer_vectors" in loaded:
                layer_vectors = loaded["layer_vectors"]
                vector = list(layer_vectors.values())[0]  # Primary vector
            # Load cluster vectors if present
            if "cluster_vectors" in loaded:
                cluster_vectors = loaded["cluster_vectors"]
        else:
            vector = loaded
        print(f"Loaded vector: shape={vector.shape}, norm={vector.norm().item():.4f}")
    
    elif args.mode in ["extract", "train", "both"]:
        print(f"\n{'='*60}")
        
        if args.method == "extracted":
            print("Extracting CoT Vector...")
            method = ExtractedCoTVector(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                layer_idx=args.layer_idx,
                dataset_type=args.dataset,
            )
            vector = method.extract(support_samples)
            
        elif args.method == "learnable":
            print("Training Learnable CoT Vector...")
            method = LearnableCoTVector(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                layer_idx=args.layer_idx,
                dataset_type=args.dataset,
                lambda_val=args.lambda_val,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_length,
            )
            vector = method.train(support_samples, wandb_run)
            
        elif args.method == "ua":
            if args.multi_layer and args.target_layers:
                # Multi-layer UA extraction
                layer_indices = [int(l.strip()) for l in args.target_layers.split(",")]
                layer_taus = None
                if args.layer_weights:
                    # Reuse layer_weights as tau values for simplicity
                    layer_taus = [float(w.strip()) * args.tau_squared for w in args.layer_weights.split(",")]
                
                print(f"Extracting Multi-Layer UA Vectors at layers {layer_indices}...")
                method = MultiLayerUAVector(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    layer_indices=layer_indices,
                    dataset_type=args.dataset,
                    tau_squared=args.tau_squared,
                    layer_taus=layer_taus,
                    min_variance=args.min_variance,
                )
                layer_vectors = method.extract(support_samples)
                vector = method.get_vector()  # Primary vector
            else:
                # Single-layer UA extraction
                print("Extracting Uncertainty-Aware CoT Vector...")
                method = UACoTVector(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    layer_idx=args.layer_idx,
                    dataset_type=args.dataset,
                    tau_squared=args.tau_squared,
                    min_variance=args.min_variance,
                )
                vector = method.extract(support_samples)
        
        elif args.method == "mixture_ua":
            print("Extracting Mixture UA CoT Vector (IGMM)...")
            method = MixtureUACoTVector(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                layer_idx=args.layer_idx,
                dataset_type=args.dataset,
                tau_squared=args.tau_squared,
                min_variance=args.min_variance,
                num_components=args.num_mixture_components,
                concentration_prior=args.mixture_concentration,
            )
            vector = method.extract(support_samples)
            # Get all cluster vectors for saving
            cluster_vectors = method.get_all_cluster_vectors()
        
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # Save vector
        if args.save_vector and vector is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vector_filename = f"{args.method}_{args.dataset}_L{args.layer_idx}_{timestamp}.pt"
            vector_path = os.path.join(args.output_dir, vector_filename)
            
            # Prepare save data
            save_data = {
                "vector": vector.cpu(),
                "args": vars(args),
                "method": args.method,
            }
            
            # Include additional statistics for UA method
            if args.method == "ua" and hasattr(method, 'get_statistics'):
                save_data["statistics"] = method.get_statistics()
            
            # Include layer vectors for multi-layer
            if layer_vectors is not None:
                save_data["layer_vectors"] = {k: v.cpu() for k, v in layer_vectors.items()}
            
            # Include ALL cluster vectors for mixture_ua method
            if args.method == "mixture_ua" and cluster_vectors is not None:
                save_data["cluster_vectors"] = {k: v.cpu() for k, v in cluster_vectors.items()}
                # Also include full statistics for analysis
                if hasattr(method, 'get_statistics'):
                    save_data["statistics"] = method.get_statistics()
            
            torch.save(save_data, vector_path)
            print(f"Vector saved to {vector_path}")
            
            # Print additional info for mixture_ua
            if args.method == "mixture_ua" and cluster_vectors:
                print(f"  Saved {len(cluster_vectors)} cluster vectors")
    
    # Evaluation
    if args.mode in ["eval", "both"] and test_samples:
        print(f"\n{'='*60}")
        print("Evaluation")
        print("=" * 60)
        
        # Baseline evaluation
        baseline_results = None
        if not args.skip_baseline:
            print("\n[1/2] Baseline (no injection)...")
            baseline_results = run_baseline_evaluation(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                test_samples=test_samples,
                dataset_type=args.dataset,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
            )
        
        # Injection evaluation
        injection_results = None
        if vector is not None:
            if layer_vectors is not None and args.multi_layer:
                # Multi-layer evaluation
                print(f"\n[2/2] With Multi-Layer UA Vectors...")
                
                # Parse layer weights if provided
                layer_weights = None
                if args.layer_weights:
                    layer_indices = [int(l.strip()) for l in args.target_layers.split(",")]
                    weights = [float(w.strip()) for w in args.layer_weights.split(",")]
                    layer_weights = dict(zip(layer_indices, weights))
                
                # Use multi-layer evaluator
                evaluator = MultiLayerEvaluator(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    layer_vectors=layer_vectors,
                    layer_weights=layer_weights,
                )
                evaluator.register_all_hooks()
                
                # Run evaluation with hooks registered
                from src.eval import CoTEvaluator
                cot_evaluator = CoTEvaluator(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    dataset_type=args.dataset,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    use_early_stopping=args.use_early_stopping,
                )
                injection_results = cot_evaluator.evaluate_dataset(
                    test_samples,
                    desc="Multi-Layer Injection"
                )
                evaluator.clear_hooks()
            else:
                # Single-layer evaluation (works for extracted, learnable, ua, mixture_ua)
                print(f"\n[2/2] With CoT Vector (layer {args.layer_idx})...")
                injection_results = run_injection_evaluation(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    test_samples=test_samples,
                    vector=vector,
                    layer_idx=args.layer_idx,
                    dataset_type=args.dataset,
                    scaling_factor=args.scaling_factor,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    use_early_stopping=args.use_early_stopping,
                )
        
        # Print results
        print("\n" + "=" * 60)
        print("Results Summary")
        print("-" * 60)
        print(f"Model:      {args.model_path.split('/')[-1]}")
        print(f"Method:     {args.method}")
        print(f"Layer:      {args.layer_idx}" + (f" (multi: {args.target_layers})" if args.multi_layer else ""))
        print(f"Dataset:    {args.dataset}")
        print(f"Test size:  {len(test_samples)}")
        print("-" * 60)
        
        if baseline_results:
            print(f"Baseline:   {baseline_results['accuracy']:.2f}% ({baseline_results['correct']}/{baseline_results['total']})")
        
        if injection_results:
            if baseline_results:
                diff = injection_results['accuracy'] - baseline_results['accuracy']
                sign = "+" if diff >= 0 else ""
                print(f"Injection:  {injection_results['accuracy']:.2f}% ({injection_results['correct']}/{injection_results['total']}) [{sign}{diff:.2f}%]")
            else:
                print(f"Injection:  {injection_results['accuracy']:.2f}% ({injection_results['correct']}/{injection_results['total']})")
        
        if vector is not None:
            print(f"Vec norm:   {vector.norm().item():.4f}")
        
        # Print mixture_ua specific info
        if args.method == "mixture_ua" and cluster_vectors:
            print(f"Clusters:   {len(cluster_vectors)} valid clusters discovered")
        
        print("=" * 60)
        
        # Log to WandB
        if wandb_run:
            if baseline_results:
                wandb_run.log({
                    "eval/baseline_accuracy": baseline_results['accuracy'],
                })
            if injection_results:
                log_dict = {
                    "eval/injection_accuracy": injection_results['accuracy'],
                    "eval/vector_norm": vector.norm().item() if vector is not None else 0,
                }
                if baseline_results:
                    log_dict["eval/improvement"] = injection_results['accuracy'] - baseline_results['accuracy']
                # Add mixture_ua specific metrics
                if args.method == "mixture_ua" and cluster_vectors:
                    log_dict["eval/num_clusters"] = len(cluster_vectors)
                wandb_run.log(log_dict)
            wandb_run.finish()
    
    print("\nDone!")


if __name__ == "__main__":
    main()