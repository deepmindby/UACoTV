import os
import torch
import argparse
import glob
import numpy as np
from tabulate import tabulate

def analyze_single_vector(file_path):
    """
    分析单个 .pt 文件的统计信息
    """
    try:
        data = torch.load(file_path, map_location="cpu")
    except Exception as e:
        return None, f"Error loading: {str(e)}"

    # 检查是否为 UA 方法且包含统计信息
    if data.get("method") not in ["ua", "mixture_ua"]: 
        return None, "Not a UA vector file"
    
    stats = data.get("statistics", {})
    if not stats:
        return None, "No statistics found in file"

    # 获取核心数据
    shrinkage_coeffs = stats.get("shrinkage_coefficients") # Lambda
    tau_squared = stats.get("tau_squared", 1.0)
    mean_vec = stats.get("mean_vector")
    ua_vec = stats.get("ua_vector")
    
    if shrinkage_coeffs is None:
        return None, "Missing shrinkage coefficients"

    # 转换为 numpy
    coeffs = shrinkage_coeffs.numpy()
    
    # 计算指标
    mean_lambda = np.mean(coeffs)
    pct_preserved = np.sum(coeffs > 0.8) / len(coeffs) * 100 # 认为是信号
    pct_suppressed = np.sum(coeffs < 0.2) / len(coeffs) * 100 # 认为是噪声
    
    norm_ratio = 0.0
    if mean_vec is not None and ua_vec is not None:
        norm_orig = torch.norm(mean_vec).item()
        norm_ua = torch.norm(ua_vec).item()
        if norm_orig > 1e-9:
            norm_ratio = norm_ua / norm_orig

    # 给出建议
    suggestion = "Keep"
    if pct_preserved > 95:
        suggestion = "⬇️ Decrease tau² (Too little denoising)"
    elif pct_suppressed > 95:
        suggestion = "⬆️ Increase tau² (Signal loss risk)"
    elif pct_preserved < 5 and pct_suppressed < 95:
        # 这种情况比较少见，可能是过度平滑但没完全抑制
        suggestion = "⬆️ Increase tau² (Over-regularized)"
    else:
        suggestion = "✅ Balanced (Fine-tune)"

    # 从文件名解析 Layer
    filename = os.path.basename(file_path)
    try:
        layer_str = filename.split("_L")[1].split(".")[0].split("_")[0]
        layer_idx = int(layer_str)
    except:
        layer_idx = -1

    # 【修复点】：键名必须与 main 函数中的 headers 完全一致
    return {
        "Layer": layer_idx,
        "Tau^2": tau_squared,
        "Mean λ": mean_lambda,
        "Kept%": pct_preserved,  # 修复：去掉 (>0.8) 后缀
        "Cut%": pct_suppressed,  # 修复：去掉 (<0.2) 后缀
        "Norm Ratio": norm_ratio,
        "Suggestion": suggestion,
        "File": filename
    }, None

def main():
    parser = argparse.ArgumentParser(description="Analyze UA CoT Vectors and suggest hyperparameters.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k, math_hard)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory containing vectors")
    args = parser.parse_args()

    # 搜索文件
    search_pattern = os.path.join(args.output_dir, f"ua_{args.dataset}_L*.pt")
    files = glob.glob(search_pattern)
    files.sort() 

    if not files:
        print(f"No UA vector files found for dataset '{args.dataset}' in {args.output_dir}")
        return

    print(f"\nAnalyzing {len(files)} vectors for dataset: {args.dataset}\n")

    results = []
    for f in files:
        res, err = analyze_single_vector(f)
        if res:
            results.append(res)
        else:
            if "Not a UA vector" not in err: 
                print(f"Skipping {os.path.basename(f)}: {err}")

    # 按层号排序
    results.sort(key=lambda x: x["Layer"])

    # 打印表格
    if results:
        headers = ["Layer", "Tau^2", "Mean λ", "Kept%", "Cut%", "Norm Ratio", "Suggestion"]
        
        # 构建表格数据
        table_data = []
        for r in results:
            row = []
            for h in headers:
                val = r.get(h, "N/A") # 使用 get 防止 KeyError
                if h == "Kept%" or h == "Cut%":
                    row.append(f"{val:.1f}%")
                elif h == "Mean λ" or h == "Norm Ratio":
                    row.append(f"{val:.3f}")
                elif h == "Tau^2":
                    row.append(f"{val:.2f}")
                else:
                    row.append(val)
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="simple_grid"))
        
        print("\n" + "="*60)
        print("💡 Guide to Adjustment:")
        print("  • Kept%: Dimensions with λ > 0.8 (High Confidence).")
        print("  • Cut%:  Dimensions with λ < 0.2 (Noise).")
        print("  • Rule:  If Kept% > 95% -> Decrease tau^2 (0.1, 0.01)")
        print("           If Cut% > 95%  -> Increase tau^2 (2.0, 5.0)")
        print("="*60 + "\n")
    else:
        print("No valid UA vectors found to analyze.")

if __name__ == "__main__":
    main()