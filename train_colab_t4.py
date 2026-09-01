# ==============================================================================
# 🚀 QU-SSM-MoE: Google Colab T4 Fast-Fluency Distillation Pipeline
# Optimized for 16GB T4 GPU with Mixed Precision (FP16) & Streaming Datasets
# ==============================================================================

# Step 1: Install Dependencies (Run this in Colab)
# !pip install torch transformers datasets accelerate huggingface_hub -q

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
from huggingface_hub import HfApi

# ------------------------------------------------------------------------------
# 1. ARCHITECTURE DEFINITION (Self-Contained for Colab)
# ------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class StaticTPUMoE(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 1024, num_experts: int = 8, moe_top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.moe_top_k = min(moe_top_k, num_experts)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        router_logits = self.router(x_flat) * (1.0 / math.sqrt(self.d_model))
        probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, self.moe_top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        sparse_weights = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_weights)
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_weights = sparse_weights[..., i:i+1]
            if expert_weights.any():
                out = out + expert_weights * expert(x_flat)
        return out.view(*orig_shape)

class ExactRealQUBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_state: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.theta_proj = nn.Linear(d_model, d_model * d_state, bias=False)
        self.gamma_proj = nn.Linear(d_model, d_model, bias=True)
        self.u_proj = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model * d_state, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.d_val = nn.Parameter(torch.ones(d_model))
        self.theta_bias = nn.Parameter(torch.linspace(0.01, 0.5, d_model * d_state))

    def forward(self, x):
        B, L, D = x.shape
        N = self.d_state
        theta = (self.theta_proj(x) + self.theta_bias).view(B, L, D, N)
        log_g = F.logsigmoid(self.gamma_proj(x)).unsqueeze(-1).expand(B, L, D, N)
        u_val = self.u_proj(x)
        u = u_val.unsqueeze(-1).expand(B, L, D, N)
        S = torch.cumsum(log_g, dim=1).clamp(min=-12.0, max=0.0)
        Phi = torch.cumsum(theta, dim=1)
        exp_S = torch.exp(S)
        exp_neg_S = torch.exp(-S)
        cos_Phi = torch.cos(Phi)
        sin_Phi = torch.sin(Phi)
        u_scaled_real = u * exp_neg_S * cos_Phi
        u_scaled_imag = -u * exp_neg_S * sin_Phi
        cum_real = torch.cumsum(u_scaled_real, dim=1)
        cum_imag = torch.cumsum(u_scaled_imag, dim=1)
        h_exact = exp_S * (cos_Phi * cum_real - sin_Phi * cum_imag)
        h_flat = h_exact.contiguous().view(B, L, D * N)
        y_ssm = self.c_proj(h_flat) + x * self.d_val
        return y_ssm * F.silu(self.gate_proj(x))

class QUSSMModel(nn.Module):
    def __init__(self, d_model=512, n_layers=6, d_state=8, d_ff=1024, num_experts=8, moe_top_k=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ssm_norm": RMSNorm(d_model),
                "ssm": ExactRealQUBlock(d_model=d_model, d_state=d_state),
                "moe_norm": RMSNorm(d_model),
                "moe": StaticTPUMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, moe_top_k=moe_top_k)
            })
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer["ssm"](layer["ssm_norm"](x))
            x = x + layer["moe"](layer["moe_norm"](x))
        return self.final_norm(x)

class QUSSMForCausalLM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_layers=6, d_state=8, d_ff=1024, num_experts=8, moe_top_k=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = QUSSMModel(d_model, n_layers, d_state, d_ff, num_experts, moe_top_k)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        h = self.backbone(x)
        return self.lm_head(h)

    def generate(self, input_ids, max_new_tokens=40, temperature=0.7, top_k=50):
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# ------------------------------------------------------------------------------
# 2. T4 COLAB DISTILLATION RUNNER
# ------------------------------------------------------------------------------
def run_colab_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[✓] Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Initialize Teacher & Tokenizer
    teacher_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    print(f"[*] Loading Teacher: {teacher_id} in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Initialize Student (350M-MoE Tier)
    print("[*] Initializing Student: QU-SSM-350M-MoE (159M Active Params)...")
    student = QUSSMForCausalLM(
        vocab_size=50257,
        d_model=640,
        n_layers=10,
        d_state=8,
        d_ff=1280,
        num_experts=8,
        moe_top_k=2
    ).to(device)

    total_p = sum(p.numel() for p in student.parameters())
    print(f"[✓] Student Initialized: {total_p / 1e6:.2f}M Total Parameters")

    # 3. Optimizer, Scheduler & Mixed Precision Scaler
    optimizer = torch.optim.AdamW(student.parameters(), lr=6e-4, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=(device == "cuda"))

    # 4. Stream Dataset (Cosmopedia-v2 Synthetic Reasoning)
    print("[*] Streaming high-quality educational dataset...")
    dataset = load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)

    # 5. Training Loop
    student.train()
    batch_size = 8
    seq_len = 512
    grad_accum_steps = 4  # Effective batch size = 32 tokens (16,384 tokens/step)
    target_steps = 3000   # ~50M Tokens total (Takes ~2.5 hours on T4)

    print("\n" + "="*70)
    print("🚀 TRAINING STARTED (Live Generation Every 300 Steps)")
    print("="*70 + "\n")

    step = 0
    running_loss = 0.0
    start_time = time.time()
    accum_tokens = []

    for item in dataset:
        text = item.get("text", "")
        if len(text) < 100:
            continue

        tokens = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt").input_ids[0]
        if len(tokens) < seq_len:
            continue
        accum_tokens.append(tokens)

        if len(accum_tokens) == batch_size:
            input_ids = torch.stack(accum_tokens).to(device)
            accum_tokens = []

            # Mixed Precision Forward Pass
            with autocast(enabled=(device == "cuda")):
                # Teacher Soft Logits
                with torch.no_grad():
                    t_logits = teacher(input_ids).logits
                
                # Student Logits
                s_logits = student(input_ids)

                v_min = min(s_logits.size(-1), t_logits.size(-1))
                s_shift = s_logits[..., :-1, :v_min].contiguous().view(-1, v_min)
                t_shift = t_logits[..., :-1, :v_min].contiguous().view(-1, v_min)
                labels = input_ids[..., 1:].contiguous().view(-1)

                # Combined Loss: 50% Hard Ground Truth + 50% Soft Teacher Probabilities
                T = 2.0
                ce_loss = F.cross_entropy(s_shift, labels)
                kl_loss = F.kl_div(
                    F.log_softmax(s_shift / T, dim=-1),
                    F.softmax(t_shift / T, dim=-1),
                    reduction="batchmean"
                ) * (T ** 2)
                loss = (0.5 * ce_loss + 0.5 * kl_loss) / grad_accum_steps

            # Backward with Scaler
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps
            step += 1

            # Logging & Live Generation Test
            if step % 50 == 0:
                elapsed = time.time() - start_time
                tok_per_sec = (step * batch_size * seq_len) / elapsed
                avg_loss = running_loss / 50
                running_loss = 0.0
                print(f"Step [{step:4d}/{target_steps}] | Loss: {avg_loss:.4f} | Speed: {tok_per_sec:.0f} tok/s | Elapsed: {elapsed/60:.1f}m")

            if step % 300 == 0 or step == target_steps:
                print("\n" + "-"*50)
                student.eval()
                sample_prompt = "The fundamental law of physics states that"
                prompt_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    gen_ids = student.generate(prompt_ids, max_new_tokens=35, temperature=0.7)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                print(f"🤖 [LIVE GENERATION @ Step {step}]:\n{gen_text}")
                print("-"*50 + "\n")
                student.train()

                # Save checkpoint
                torch.save(student.state_dict(), f"qu_ssm_350m_step_{step}.pt")
                print(f"💾 Checkpoint saved: qu_ssm_350m_step_{step}.pt\n")

            if step >= target_steps:
                break

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("Final Model Checkpoint: qu_ssm_350m_final.pt")
    print("="*70)
    torch.save(student.state_dict(), "qu_ssm_350m_final.pt")

if __name__ == "__main__":
    run_colab_training()
