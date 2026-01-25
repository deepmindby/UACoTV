# CoT Vectors: Variational Framework

Refactored codebase for Chain-of-Thought Vectors based on the **Variational CoT Vectors** framework.

## Key Changes from Original

**Removed:**
- All RL-related code (GRPO, DAPO, Self-Evolved methods)
- `src/rl_solvers/` directory
- RL-specific arguments and logic paths

**Preserved:**
- `extracted`: Statistical aggregation method (Eq. 4-5)
- `learnable`: Teacher-student gradient optimization (Eq. 6)

**Added:**
- `ua`: Uncertainty-Aware method with Bayesian shrinkage
- Multi-layer UA injection support

## Mathematical Foundation

### Extracted Method
Simple mean aggregation:
```
v_E = (1/N) Σ v_i
```

### Learnable Method
Teacher-student optimization:
```
L = L_align + λ * L_CE
```

### UA Method (New)
Bayesian MAP estimation with structured prior:
```
Prior:      p(z) = N(0, τ²I)
Likelihood: p(μ|z) = N(z, σ²)
Posterior:  z_d = k_d * μ_d
            where k_d = σ²_d / (σ²_d + τ²)
```

The shrinkage coefficient `k_d` acts as adaptive gating:
- High variance (noise) → k_d → 0 (suppress)
- Low variance (signal) → k_d → 1 (preserve)

## Usage

### Single-Layer Extracted
```bash
python main.py \
    --method extracted \
    --layer_idx 10 \
    --dataset gsm8k \
    --model_path /path/to/model
```

### Single-Layer UA
```bash
python main.py \
    --method ua \
    --layer_idx 10 \
    --tau_squared 1.0 \
    --dataset gsm8k
```

### Multi-Layer UA
```bash
python main.py \
    --method ua \
    --multi_layer \
    --target_layers "0,5,10,15" \
    --layer_weights "1.0,0.8,0.6,0.4" \
    --tau_squared 1.0
```

### Layer Sweep
```bash
python run_layer_sweep.py \
    --method ua \
    --layers "0,5,10,15,20,25" \
    --tau_squared 1.0 \
    --save_vectors
```

## Project Structure

```
cot_vectors_refactored/
├── main.py                 # Main entry point
├── run_layer_sweep.py      # Layer sweep script
├── requirements.txt        # Dependencies (no RL libs)
└── src/
    ├── __init__.py
    ├── args.py             # Arguments (RL args removed)
    ├── data_utils.py       # Data loading
    ├── eval.py             # Evaluation
    ├── models.py           # Model wrapper with hooks
    ├── utils.py            # Utilities
    └── methods/
        ├── __init__.py     # Method exports
        ├── base.py         # Base class
        ├── extracted.py    # Extracted method
        ├── learnable.py    # Learnable method
        ├── ua_vector.py    # UA method (NEW)
        └── multi_layer_ua.py  # Multi-layer UA (NEW)
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--method` | `extracted`, `learnable`, `ua` | `extracted` |
| `--layer_idx` | Target layer for injection | `0` |
| `--tau_squared` | Prior variance τ² for UA | `1.0` |
| `--min_variance` | Minimum variance threshold | `1e-6` |
| `--multi_layer` | Enable multi-layer injection | `False` |
| `--target_layers` | Comma-separated layer indices | `None` |
| `--layer_weights` | Comma-separated scaling weights | `None` |

## Output

### Vector Statistics (UA Method)
The UA method provides detailed statistics:
- Mean vector norm
- UA vector norm
- Shrinkage ratio
- Dimension classification (highly suppressed, moderately kept, well preserved)

### Saved Vector Format
```python
{
    "vector": tensor,           # The CoT vector
    "args": {...},              # Arguments used
    "method": "ua",             # Method name
    "statistics": {             # For UA method only
        "mean_vector": tensor,
        "variance_vector": tensor,
        "shrinkage_coefficients": tensor,
        "tau_squared": float,
    },
    "layer_vectors": {...},     # For multi-layer only
}
```
