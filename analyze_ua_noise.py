#!/usr/bin/env python3
"""
Analyze UA Noise Hypothesis: "High variance dimensions represent reasoning noise"

This script validates the core assumption that dimensions with high variance
in activation differences are noise and should be suppressed.

Core Hypothesis:
    v_j* = mean_j / (1 + gamma * variance_j)
    
When variance_j is large, the dimension is suppressed (considered noise).
When variance_j is small, the dimension is preserved (considered signal).

Experiments:
A. Variance Masking: Zero out high/low variance dimensions and measure accuracy
B. Signal-to-Noise Ratio: Analyze |mean| / (variance + eps) distribution  
C. Visualization: Plot variance distribution and UA suppression zones

Usage:
    python analyze_ua_noise.py \
        --model_path /path/to/model \
        --dataset gsm8k \
        --layer 10 \
        --gamma 1.0 \
        --num_support_samples 500 \
        --num_test_samples 100
"""

import os
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset, PROMPT_TEMPLATES
from src.eval import run_injection_evaluation, run_baseline_evaluation
from src.utils import set_seed


class VectorExtractor:
    """Extract activation difference vectors from support samples."""
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
    ):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.dataset_type = dataset_type
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def extract_single(self, sample) -> torch.Tensor:
        """Extract activation difference for a single sample."""
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
        
        # Extract CoT activations
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        cot_answer_activation = cot_activation[:, cot_answer_pos, :].mean(dim=1)
        
        # Extract non-CoT activations
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        non_cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        non_cot_answer_activation = non_cot_activation[:, non_cot_answer_pos, :].mean(dim=1)
        
        # Compute difference
        diff = cot_answer_activation - non_cot_answer_activation
        
        self.model_wrapper.clear_hooks()
        
        return diff.squeeze(0)
    
    def extract_all(self, support_samples: List) -> torch.Tensor:
        """Extract vectors from all support samples."""
        vectors = []
        for sample in tqdm(support_samples, desc="Extracting vectors", ncols=100):
            try:
                vec = self.extract_single(sample)
                vectors.append(vec)
            except Exception as e:
                continue
        
        if not vectors:
            raise ValueError("No vectors extracted!")
        
        return torch.stack(vectors, dim=0)  # [N, hidden]


def compute_ua_vector(mean_vec: torch.Tensor, var_vec: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute UA vector using the hypothesis formula.
    
    v_j* = mean_j / (1 + gamma * variance_j)
    
    Args:
        mean_vec: Mean vector [hidden]
        var_vec: Variance vector [hidden]
        gamma: Scaling factor for variance suppression
    
    Returns:
        UA vector [hidden]
    """
    suppression_factor = 1.0 / (1.0 + gamma * var_vec)
    return mean_vec * suppression_factor


def experiment_a_variance_masking(
    model_wrapper: CoTModelWrapper,
    tokenizer,
    test_samples: List,
    mean_vec: torch.Tensor,
    var_vec: torch.Tensor,
    layer_idx: int,
    dataset_type: str,
    percentiles: List[int] = [10, 20, 30, 50],
    max_new_tokens: int = 512,
    num_beams: int = 3,
) -> Dict[str, Any]:
    """
    Experiment A: Variance Masking
    
    按照方差百分位生成 Mask，分别将高方差维度和低方差维度置零，
    计算其注入模型后的推理准确率。
    
    Args:
        model_wrapper: Model wrapper
        tokenizer: Tokenizer
        test_samples: Test samples
        mean_vec: Mean vector
        var_vec: Variance vector
        layer_idx: Layer index
        dataset_type: Dataset type
        percentiles: Variance percentiles to test
        max_new_tokens: Max tokens for generation
        num_beams: Number of beams
    
    Returns:
        Dictionary containing experiment results
    """
    print("\n" + "=" * 70)
    print("Experiment A: Variance Masking")
    print("=" * 70)
    
    results = {
        "percentiles": percentiles,
        "high_variance_zeroed": [],  # 高方差维度置零
        "low_variance_zeroed": [],   # 低方差维度置零
    }
    
    hidden_size = mean_vec.shape[0]
    
    # 首先运行原始均值向量的基准
    print("\n>>> Baseline: Using raw mean vector...")
    baseline_result = run_injection_evaluation(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        test_samples=test_samples,
        vector=mean_vec,
        layer_idx=layer_idx,
        dataset_type=dataset_type,
        scaling_factor=1.0,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    results["mean_vector_accuracy"] = baseline_result["accuracy"]
    print(f"  Mean vector accuracy: {baseline_result['accuracy']:.2f}%")
    
    for pct in percentiles:
        print(f"\n>>> Testing {pct}% variance threshold...")
        
        # 计算方差的百分位阈值
        threshold = torch.quantile(var_vec, pct / 100.0).item()
        
        # 高方差 mask (方差 > 阈值的维度)
        high_var_mask = var_vec > threshold
        # 低方差 mask (方差 <= 阈值的维度)
        low_var_mask = var_vec <= threshold
        
        n_high = high_var_mask.sum().item()
        n_low = low_var_mask.sum().item()
        
        print(f"  Threshold: {threshold:.6f}")
        print(f"  High variance dims: {n_high} ({100*n_high/hidden_size:.1f}%)")
        print(f"  Low variance dims: {n_low} ({100*n_low/hidden_size:.1f}%)")
        
        # 实验 A.1: 将高方差维度置零 (保留低方差信号)
        vec_high_zeroed = mean_vec.clone()
        vec_high_zeroed[high_var_mask] = 0.0
        
        result_high_zeroed = run_injection_evaluation(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            test_samples=test_samples,
            vector=vec_high_zeroed,
            layer_idx=layer_idx,
            dataset_type=dataset_type,
            scaling_factor=1.0,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        
        results["high_variance_zeroed"].append({
            "percentile": pct,
            "threshold": threshold,
            "n_zeroed": n_high,
            "accuracy": result_high_zeroed["accuracy"],
            "correct": result_high_zeroed["correct"],
            "total": result_high_zeroed["total"],
        })
        print(f"  [High var zeroed] Accuracy: {result_high_zeroed['accuracy']:.2f}%")
        
        # 实验 A.2: 将低方差维度置零 (保留高方差噪声)
        vec_low_zeroed = mean_vec.clone()
        vec_low_zeroed[low_var_mask] = 0.0
        
        result_low_zeroed = run_injection_evaluation(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            test_samples=test_samples,
            vector=vec_low_zeroed,
            layer_idx=layer_idx,
            dataset_type=dataset_type,
            scaling_factor=1.0,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        
        results["low_variance_zeroed"].append({
            "percentile": pct,
            "threshold": threshold,
            "n_zeroed": n_low,
            "accuracy": result_low_zeroed["accuracy"],
            "correct": result_low_zeroed["correct"],
            "total": result_low_zeroed["total"],
        })
        print(f"  [Low var zeroed] Accuracy: {result_low_zeroed['accuracy']:.2f}%")
        
        # 计算差异
        diff = result_high_zeroed["accuracy"] - result_low_zeroed["accuracy"]
        print(f"  Difference (high_zeroed - low_zeroed): {diff:+.2f}%")
    
    return results


def experiment_b_signal_to_noise(
    mean_vec: torch.Tensor,
    var_vec: torch.Tensor,
    gamma: float,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Experiment B: Signal-to-Noise Ratio Analysis
    
    计算该层向量的"有效信号比"，即 (mean.abs() / (variance + eps)) 的分布情况。
    
    Args:
        mean_vec: Mean vector
        var_vec: Variance vector
        gamma: Gamma parameter for UA
        eps: Small epsilon for numerical stability
    
    Returns:
        Dictionary containing SNR analysis results
    """
    print("\n" + "=" * 70)
    print("Experiment B: Signal-to-Noise Ratio Analysis")
    print("=" * 70)
    
    # 计算 SNR = |mean| / (variance + eps)
    snr = mean_vec.abs() / (var_vec + eps)
    
    # 计算 UA 抑制因子 suppression = 1 / (1 + gamma * variance)
    suppression_factor = 1.0 / (1.0 + gamma * var_vec)
    
    # 计算 UA 后的 effective signal = |mean * suppression|
    effective_signal = (mean_vec * suppression_factor).abs()
    
    # 统计信息
    hidden_size = mean_vec.shape[0]
    
    # SNR 分布统计
    snr_np = snr.cpu().numpy()
    snr_percentiles = {
        "p10": np.percentile(snr_np, 10),
        "p25": np.percentile(snr_np, 25),
        "p50": np.percentile(snr_np, 50),
        "p75": np.percentile(snr_np, 75),
        "p90": np.percentile(snr_np, 90),
    }
    
    # 分类维度
    # 高 SNR (信号强): SNR > median
    # 低 SNR (噪声多): SNR < median
    snr_median = snr_percentiles["p50"]
    high_snr_mask = snr > snr_median
    low_snr_mask = snr <= snr_median
    
    # 计算各组的 suppression factor 分布
    high_snr_suppression = suppression_factor[high_snr_mask]
    low_snr_suppression = suppression_factor[low_snr_mask]
    
    results = {
        "snr_statistics": {
            "mean": snr.mean().item(),
            "std": snr.std().item(),
            "min": snr.min().item(),
            "max": snr.max().item(),
            "percentiles": snr_percentiles,
        },
        "suppression_statistics": {
            "mean": suppression_factor.mean().item(),
            "std": suppression_factor.std().item(),
            "min": suppression_factor.min().item(),
            "max": suppression_factor.max().item(),
        },
        "high_snr_group": {
            "count": high_snr_mask.sum().item(),
            "avg_suppression": high_snr_suppression.mean().item(),
            "avg_variance": var_vec[high_snr_mask].mean().item(),
        },
        "low_snr_group": {
            "count": low_snr_mask.sum().item(),
            "avg_suppression": low_snr_suppression.mean().item(),
            "avg_variance": var_vec[low_snr_mask].mean().item(),
        },
        "correlation": {
            "snr_vs_suppression": torch.corrcoef(
                torch.stack([snr, suppression_factor])
            )[0, 1].item(),
            "variance_vs_suppression": torch.corrcoef(
                torch.stack([var_vec, suppression_factor])
            )[0, 1].item(),
        },
        "raw_data": {
            "snr": snr.cpu().numpy().tolist(),
            "suppression_factor": suppression_factor.cpu().numpy().tolist(),
            "variance": var_vec.cpu().numpy().tolist(),
            "mean_abs": mean_vec.abs().cpu().numpy().tolist(),
        }
    }
    
    # 打印结果
    print(f"\nSNR Statistics (|mean| / (var + eps)):")
    print(f"  Mean: {results['snr_statistics']['mean']:.4f}")
    print(f"  Std: {results['snr_statistics']['std']:.4f}")
    print(f"  Min: {results['snr_statistics']['min']:.6f}")
    print(f"  Max: {results['snr_statistics']['max']:.2f}")
    print(f"  Median: {snr_percentiles['p50']:.4f}")
    
    print(f"\nSuppression Factor Statistics (1 / (1 + γ·σ²), γ={gamma}):")
    print(f"  Mean: {results['suppression_statistics']['mean']:.4f}")
    print(f"  Std: {results['suppression_statistics']['std']:.4f}")
    print(f"  Min: {results['suppression_statistics']['min']:.6f}")
    print(f"  Max: {results['suppression_statistics']['max']:.4f}")
    
    print(f"\nHigh SNR Group (SNR > {snr_median:.4f}):")
    print(f"  Count: {results['high_snr_group']['count']} dims")
    print(f"  Avg suppression: {results['high_snr_group']['avg_suppression']:.4f}")
    print(f"  Avg variance: {results['high_snr_group']['avg_variance']:.6f}")
    
    print(f"\nLow SNR Group (SNR <= {snr_median:.4f}):")
    print(f"  Count: {results['low_snr_group']['count']} dims")
    print(f"  Avg suppression: {results['low_snr_group']['avg_suppression']:.4f}")
    print(f"  Avg variance: {results['low_snr_group']['avg_variance']:.6f}")
    
    print(f"\nCorrelations:")
    print(f"  SNR vs Suppression: {results['correlation']['snr_vs_suppression']:.4f}")
    print(f"  Variance vs Suppression: {results['correlation']['variance_vs_suppression']:.4f}")
    
    return results


def experiment_c_visualization(
    mean_vec: torch.Tensor,
    var_vec: torch.Tensor,
    gamma: float,
    exp_a_results: Dict[str, Any],
    exp_b_results: Dict[str, Any],
    output_dir: str,
    layer_idx: int,
    dataset: str,
) -> str:
    """
    Experiment C: Visualization
    
    绘制该层维度的方差分布直方图，并标注出被 UA 机制（gamma 参数）抑制最严重的维度区间。
    
    Args:
        mean_vec: Mean vector
        var_vec: Variance vector
        gamma: Gamma parameter
        exp_a_results: Results from Experiment A
        exp_b_results: Results from Experiment B
        output_dir: Output directory
        layer_idx: Layer index
        dataset: Dataset name
    
    Returns:
        Path to saved figure
    """
    print("\n" + "=" * 70)
    print("Experiment C: Visualization")
    print("=" * 70)
    
    # 创建大图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'UA Noise Analysis: Layer {layer_idx}, Dataset: {dataset}, γ={gamma}', 
                 fontsize=14, fontweight='bold')
    
    var_np = var_vec.cpu().numpy()
    mean_np = mean_vec.cpu().numpy()
    suppression = 1.0 / (1.0 + gamma * var_np)
    snr = np.abs(mean_np) / (var_np + 1e-6)
    
    # ==================== Plot 1: Variance Distribution ====================
    ax1 = axes[0, 0]
    
    # 绘制方差直方图
    n, bins, patches = ax1.hist(var_np, bins=50, color='steelblue', alpha=0.7, 
                                 edgecolor='black', linewidth=0.5)
    
    # 标注百分位线
    percentiles = [10, 50, 90]
    colors = ['green', 'orange', 'red']
    for pct, color in zip(percentiles, colors):
        threshold = np.percentile(var_np, pct)
        ax1.axvline(threshold, color=color, linestyle='--', linewidth=2, 
                    label=f'{pct}th percentile: {threshold:.4f}')
    
    ax1.set_xlabel('Variance (σ²)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Variance Distribution Across Dimensions', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ==================== Plot 2: Suppression Factor vs Variance ====================
    ax2 = axes[0, 1]
    
    # 散点图：方差 vs 抑制因子
    scatter = ax2.scatter(var_np, suppression, c=snr, cmap='viridis', 
                          alpha=0.5, s=2, label='Dimensions')
    
    # 理论曲线
    var_range = np.linspace(var_np.min(), var_np.max(), 100)
    theoretical_suppression = 1.0 / (1.0 + gamma * var_range)
    ax2.plot(var_range, theoretical_suppression, 'r-', linewidth=2, 
             label=f'Theoretical: 1/(1+γσ²)')
    
    # 标注抑制区间
    # 高度抑制区 (suppression < 0.2)
    high_suppress_threshold = 0.2
    var_threshold_high = (1/high_suppress_threshold - 1) / gamma if gamma > 0 else float('inf')
    ax2.axhline(high_suppress_threshold, color='red', linestyle=':', alpha=0.7, 
                label=f'Heavy suppression (<{high_suppress_threshold})')
    
    # 中度抑制区 (0.2 < suppression < 0.5)
    mid_suppress_threshold = 0.5
    ax2.axhline(mid_suppress_threshold, color='orange', linestyle=':', alpha=0.7,
                label=f'Moderate suppression (<{mid_suppress_threshold})')
    
    ax2.set_xlabel('Variance (σ²)', fontsize=11)
    ax2.set_ylabel('Suppression Factor', fontsize=11)
    ax2.set_title(f'UA Suppression vs Variance (γ={gamma})', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('SNR = |μ|/(σ²+ε)', fontsize=10)
    
    # ==================== Plot 3: Experiment A Results ====================
    ax3 = axes[1, 0]
    
    percentiles_list = exp_a_results["percentiles"]
    high_var_acc = [r["accuracy"] for r in exp_a_results["high_variance_zeroed"]]
    low_var_acc = [r["accuracy"] for r in exp_a_results["low_variance_zeroed"]]
    baseline_acc = exp_a_results["mean_vector_accuracy"]
    
    x = np.arange(len(percentiles_list))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, high_var_acc, width, label='High Var Zeroed (Keep Signal)', 
                    color='forestgreen', alpha=0.8)
    bars2 = ax3.bar(x + width/2, low_var_acc, width, label='Low Var Zeroed (Keep Noise)', 
                    color='firebrick', alpha=0.8)
    
    # 基准线
    ax3.axhline(baseline_acc, color='blue', linestyle='--', linewidth=2, 
                label=f'Mean Vector Baseline: {baseline_acc:.1f}%')
    
    ax3.set_xlabel('Variance Percentile Threshold', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Experiment A: Variance Masking Results', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{p}%' for p in percentiles_list])
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, acc in zip(bars1, high_var_acc):
        ax3.annotate(f'{acc:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar, acc in zip(bars2, low_var_acc):
        ax3.annotate(f'{acc:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # ==================== Plot 4: SNR Distribution ====================
    ax4 = axes[1, 1]
    
    # SNR 直方图
    snr_clipped = np.clip(snr, 0, np.percentile(snr, 99))  # 裁剪极端值
    n, bins, patches = ax4.hist(snr_clipped, bins=50, color='purple', alpha=0.7,
                                 edgecolor='black', linewidth=0.5)
    
    # 根据 SNR 值着色
    snr_median = np.median(snr_clipped)
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if left_edge > snr_median:
            patch.set_facecolor('forestgreen')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('firebrick')
            patch.set_alpha(0.7)
    
    ax4.axvline(snr_median, color='black', linestyle='--', linewidth=2,
                label=f'Median SNR: {snr_median:.2f}')
    
    # 添加图例说明
    signal_patch = mpatches.Patch(color='forestgreen', alpha=0.7, label='High SNR (Signal)')
    noise_patch = mpatches.Patch(color='firebrick', alpha=0.7, label='Low SNR (Noise)')
    
    ax4.set_xlabel('Signal-to-Noise Ratio (|μ|/(σ²+ε))', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('SNR Distribution (clipped at 99th percentile)', fontsize=12)
    ax4.legend(handles=[signal_patch, noise_patch, 
                        plt.Line2D([0], [0], color='black', linestyle='--', label=f'Median: {snr_median:.2f}')],
               loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(output_dir, f"ua_noise_analysis_L{layer_idx}_{dataset}_{timestamp}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {fig_path}")
    
    return fig_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze UA Noise Hypothesis: High variance dimensions represent reasoning noise"
    )
    
    # Model & Data
    parser.add_argument("--model_path", type=str, 
                        default="/home/haichao/TA/UACoTV/models/Qwen2.5-Math-7B",
                        help="Path to the pretrained model")
    parser.add_argument("--model_name", type=str, default="qwen", 
                        choices=["qwen", "llama"],
                        help="Model type")
    parser.add_argument("--data_path", type=str, 
                        default="/home/haichao/TA/UACoTV/data",
                        help="Path to the data directory")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"],
                        help="Dataset to use")
    parser.add_argument("--output_dir", type=str, default="./outputs/noise_analysis",
                        help="Directory to save outputs")
    
    # Experiment Configuration
    parser.add_argument("--layer", type=int, default=10,
                        help="Target layer index for analysis")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma parameter for UA suppression: v* = mean / (1 + gamma * var)")
    parser.add_argument("--num_support_samples", type=int, default=500,
                        help="Number of support samples for vector extraction")
    parser.add_argument("--num_test_samples", type=int, default=100,
                        help="Number of test samples for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Generation Configuration
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=3,
                        help="Number of beams for generation")
    
    # Experiment Selection
    parser.add_argument("--skip_exp_a", action="store_true", default=False,
                        help="Skip Experiment A (Variance Masking)")
    parser.add_argument("--skip_exp_b", action="store_true", default=False,
                        help="Skip Experiment B (SNR Analysis)")
    parser.add_argument("--skip_exp_c", action="store_true", default=False,
                        help="Skip Experiment C (Visualization)")
    
    # Variance Masking Percentiles
    parser.add_argument("--percentiles", type=str, default="10,20,30,50",
                        help="Comma-separated variance percentiles for Experiment A")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    percentiles = [int(p.strip()) for p in args.percentiles.split(",")]
    
    # Print configuration
    print("=" * 70)
    print("UA Noise Hypothesis Analysis")
    print("=" * 70)
    print(f"Model:         {args.model_path.split('/')[-1]}")
    print(f"Dataset:       {args.dataset}")
    print(f"Target Layer:  {args.layer}")
    print(f"Gamma (γ):     {args.gamma}")
    print(f"Support:       {args.num_support_samples} samples")
    print(f"Test:          {args.num_test_samples} samples")
    print(f"Percentiles:   {percentiles}")
    print("=" * 70)
    
    # Load model
    print("\n[1/5] Loading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    print(f"Model loaded: {model_wrapper.num_layers} layers, hidden_size={model_wrapper.hidden_size}")
    
    # Load data
    print("\n[2/5] Loading data...")
    support_samples = load_dataset(args.data_path, args.dataset, "train", args.num_support_samples)
    test_samples = load_dataset(args.data_path, args.dataset, "test", args.num_test_samples)
    print(f"Support set: {len(support_samples)} samples")
    print(f"Test set: {len(test_samples)} samples")
    
    # Extract vectors
    print("\n[3/5] Extracting activation difference vectors...")
    extractor = VectorExtractor(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        layer_idx=args.layer,
        dataset_type=args.dataset,
    )
    vectors = extractor.extract_all(support_samples)
    print(f"Extracted {vectors.shape[0]} vectors, hidden_size={vectors.shape[1]}")
    
    # Compute statistics
    mean_vec = vectors.mean(dim=0)
    var_vec = vectors.var(dim=0, unbiased=True)
    var_vec = torch.clamp(var_vec, min=1e-6)
    
    print(f"\nVector Statistics:")
    print(f"  Mean norm: {mean_vec.norm().item():.4f}")
    print(f"  Variance: mean={var_vec.mean().item():.6f}, max={var_vec.max().item():.4f}")
    
    # Initialize results
    all_results = {
        "config": vars(args),
        "vector_stats": {
            "num_vectors": vectors.shape[0],
            "hidden_size": vectors.shape[1],
            "mean_norm": mean_vec.norm().item(),
            "var_mean": var_vec.mean().item(),
            "var_max": var_vec.max().item(),
        },
        "experiments": {}
    }
    
    # Run experiments
    print("\n[4/5] Running experiments...")
    
    # Experiment A: Variance Masking
    exp_a_results = None
    if not args.skip_exp_a:
        exp_a_results = experiment_a_variance_masking(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            test_samples=test_samples,
            mean_vec=mean_vec,
            var_vec=var_vec,
            layer_idx=args.layer,
            dataset_type=args.dataset,
            percentiles=percentiles,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        all_results["experiments"]["variance_masking"] = exp_a_results
    
    # Experiment B: Signal-to-Noise Ratio
    exp_b_results = None
    if not args.skip_exp_b:
        exp_b_results = experiment_b_signal_to_noise(
            mean_vec=mean_vec,
            var_vec=var_vec,
            gamma=args.gamma,
        )
        # 移除 raw_data 以减小 JSON 文件大小
        exp_b_summary = {k: v for k, v in exp_b_results.items() if k != "raw_data"}
        all_results["experiments"]["signal_to_noise"] = exp_b_summary
    
    # Experiment C: Visualization
    fig_path = None
    if not args.skip_exp_c and exp_a_results and exp_b_results:
        fig_path = experiment_c_visualization(
            mean_vec=mean_vec,
            var_vec=var_vec,
            gamma=args.gamma,
            exp_a_results=exp_a_results,
            exp_b_results=exp_b_results,
            output_dir=args.output_dir,
            layer_idx=args.layer,
            dataset=args.dataset,
        )
        all_results["visualization_path"] = fig_path
    
    # Save results
    print("\n[5/5] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = os.path.join(args.output_dir, f"ua_noise_results_L{args.layer}_{args.dataset}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {json_path}")
    
    # Save CSV summary for Experiment A
    if exp_a_results:
        csv_path = os.path.join(args.output_dir, f"exp_a_summary_L{args.layer}_{args.dataset}_{timestamp}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("percentile,high_var_zeroed_acc,low_var_zeroed_acc,difference,baseline_acc\n")
            baseline = exp_a_results["mean_vector_accuracy"]
            for high_res, low_res in zip(exp_a_results["high_variance_zeroed"], 
                                          exp_a_results["low_variance_zeroed"]):
                diff = high_res["accuracy"] - low_res["accuracy"]
                f.write(f"{high_res['percentile']},{high_res['accuracy']:.2f},{low_res['accuracy']:.2f},{diff:.2f},{baseline:.2f}\n")
        print(f"CSV summary saved to: {csv_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    if exp_a_results:
        print("\n📊 Experiment A: Variance Masking")
        print(f"  Baseline (mean vector): {exp_a_results['mean_vector_accuracy']:.2f}%")
        print("\n  Percentile | High Var Zeroed | Low Var Zeroed | Difference")
        print("  " + "-" * 60)
        for high_res, low_res in zip(exp_a_results["high_variance_zeroed"], 
                                      exp_a_results["low_variance_zeroed"]):
            diff = high_res["accuracy"] - low_res["accuracy"]
            indicator = "✅" if diff > 0 else "❌"
            print(f"  {high_res['percentile']:3d}%       | {high_res['accuracy']:6.2f}%         | {low_res['accuracy']:6.2f}%         | {diff:+6.2f}% {indicator}")
        
        # 计算假设验证结果
        positive_diffs = sum(1 for h, l in zip(exp_a_results["high_variance_zeroed"], 
                                                exp_a_results["low_variance_zeroed"]) 
                            if h["accuracy"] > l["accuracy"])
        total_tests = len(exp_a_results["high_variance_zeroed"])
        
        print(f"\n  假设验证: {positive_diffs}/{total_tests} 个测试支持 '高方差=噪声' 假设")
        if positive_diffs == total_tests:
            print("  ✅ 假设得到完全验证：去除高方差维度总是优于去除低方差维度")
        elif positive_diffs > total_tests / 2:
            print("  ⚠️ 假设部分验证：多数测试支持，但存在例外")
        else:
            print("  ❌ 假设未得到验证：高方差维度可能包含有用信息")
    
    if exp_b_results:
        print("\n📈 Experiment B: Signal-to-Noise Ratio")
        print(f"  Correlation (Variance vs Suppression): {exp_b_results['correlation']['variance_vs_suppression']:.4f}")
        print(f"  High SNR group avg suppression: {exp_b_results['high_snr_group']['avg_suppression']:.4f}")
        print(f"  Low SNR group avg suppression: {exp_b_results['low_snr_group']['avg_suppression']:.4f}")
    
    if fig_path:
        print(f"\n📊 Visualization saved to: {fig_path}")
    
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()