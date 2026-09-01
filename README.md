# 🏛️ qu_ssm (QU-SSM-MoE): Continuous Quasi-Unitary Lie-Group State Space Models with Sparse Mixture-of-Experts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22217820.svg)](https://doi.org/10.5281/zenodo.22217820)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-qu__ssm--130Moe-orange)](https://huggingface.co/Prannesshkva/QU-SSM-130M-MoE)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-qu__ssm--Studio-blue)](https://huggingface.co/spaces/Prannesshkva/QU-SSM-Studio)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org)

Official PyTorch implementation of **qu_ssm** (**QU-SSM-MoE**), an ultra-efficient sequence modeling architecture co-designed with sparse SwiGLU Mixture-of-Experts.

**Sole Architect & Inventor**: **Prannessh K.V.A.**  
**Research DOI**: [**`10.5281/zenodo.22217820`**](https://doi.org/10.5281/zenodo.22217820)

---

## 🚀 Key Highlights & Benchmarks

* ⚡ **3.32x Generation Speedup**: Generates tokens at `5.55 tok/s` compared to `1.67 tok/s` for modern SOTA Transformers (SmolLM-135M).
* 💾 **Constant O(1) Memory**: Strictly `0.19 MB` state RAM footprint (1,894x smaller than a 360 MB KV-cache at L = 8,192).
* 🧠 **42% Lower Active Compute**: Evaluates only `78.27M active parameters` per token via Top-2 SwiGLU expert routing.
* 🌊 **Non-Dissipative Unitary Physics**: Eliminates the monotonic exponential dissipation (**e^(-α·t) → 0**) of classical real-valued SSMs via Lie-group rotations over **SO(2)** with **‖R(θ)‖₂ ≡ 1.00000**.
* 🌐 **Universal Multimodal Backbone**: Turnkey native support for Language, 16kHz Raw Audio DSP, High-Frequency Financial Ticks, and 2D Spatial Vision.

---

## 🏆 SOTA Inference Benchmark (~135M Scale)

| Model Architecture | Total Params | Active Params / Token | Generation Speed | Step Latency | RAM at L=8,192 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SmolLM-135M** (Hugging Face) | 134.52M | 134.52M (Dense) | 1.67 tok/s | 597.86 ms | 360.00 MB |
| **Mamba-130M-HF** (Albert Gu et al.) | 129.14M | 129.14M (Dense) | 1.98 tok/s | 506.18 ms | 0.19 MB |
| **[qu_ssm-130Moe](https://huggingface.co/Prannesshkva/QU-SSM-130M-MoE)** | **134.89M** | **78.27M (Sparse Top-2)** | **5.55 tok/s (🥇 3.32x)** | **180.16 ms** | **0.19 MB (🥇 Constant)** |

---

## 📦 Hugging Face Pretrained Checkpoints

Download and test the official checkpoints directly from Hugging Face:

| Model Tier | Parameters | Active Parameters | Hugging Face Repository Link |
| :--- | :---: | :---: | :--- |
| 🚀 **qu_ssm-130Moe (Flagship)** | **134.89M** | **78.27M** | [**`Prannesshkva/QU-SSM-130M-MoE`**](https://huggingface.co/Prannesshkva/QU-SSM-130M-MoE) |
| ⚡ **qu_ssm-60Moe (Mid-Tier)** | **64.30M** | **44.64M** | [**`Prannesshkva/QU-SSM-60M-MoE`**](https://huggingface.co/Prannesshkva/QU-SSM-60M-MoE) |
| 🔬 **qu_ssm-15M (Foundation)** | **29.80M** | **29.80M** | [**`Prannesshkva/QU-SSM-15M`**](https://huggingface.co/Prannesshkva/QU-SSM-15M) |
| 🎨 **qu_ssm Studio Space** | **Interactive App** | **Live Demo** | [**`Prannesshkva/QU-SSM-Studio`**](https://huggingface.co/spaces/Prannesshkva/QU-SSM-Studio) |

---

## 💻 Quickstart Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load qu_ssm-130Moe directly from Hugging Face Hub
model_id = "Prannesshkva/QU-SSM-130M-MoE"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "In a world governed by state space dynamics, the continuous rotation"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=40)
print(tokenizer.decode(output[0]))
```

---

## 🔬 Mathematical Recurrence Engine

The recurrent state update at step t evolves as a continuous unitary rotation modulated by an independent forget gate:

```text
h_t = γ_t · R(θ_t) · h_{t-1} + u_t
```

### 1. Dynamic Lie-Group Phase Angle (SO(2))
```text
θ_t = W_θ · x_t + θ_bias
R(θ_t) ∈ SO(2) with ‖R(θ_t)‖₂ ≡ 1.00000
```

### 2. Exact Real Dual-Component Parallel Prefix Scan
```text
S = cumsum(log γ_t).clamp(min=-12.0, max=0.0)
Φ = cumsum(θ_t)
h_t = exp(S) · [ cos(Φ) · cumsum(u_real) - sin(Φ) · cumsum(u_imag) ]
```

---

## 🔒 Citation, Legal License & Contact

* **Sole Architect & Inventor**: **Prannessh K.V.A.**
* **Permanent DOI**: [**`10.5281/zenodo.22217820`**](https://doi.org/10.5281/zenodo.22217820)
* **Non-Commercial Community License**: **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0)**
* **Enterprise & Commercial Inquiries**: Contact **Prannessh K.V.A.** via GitHub profile or Hugging Face.

```bibtex
@software{prannesshkva_qu_ssm_2026,
  author       = {Prannessh K.V.A.},
  title        = {QU-SSM-MoE: Continuous Quasi-Unitary Lie-Group State Space Models with Sparse Mixture-of-Experts},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22217820},
  url          = {https://doi.org/10.5281/zenodo.22217820}
}
```
