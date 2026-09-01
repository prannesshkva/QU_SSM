import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset

# Add current dir to import QU-SSM classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configuration_qu_ssm import QUSSMConfig
from modeling_qu_ssm import QUSSMForCausalLM

def create_scaled_config(tier: str = "350M") -> QUSSMConfig:
    configs = {
        "135M": dict(d_model=512, n_layers=6, d_state=8, d_ff=1024, num_experts=8, moe_top_k=2),
        "350M": dict(d_model=640, n_layers=10, d_state=8, d_ff=1280, num_experts=8, moe_top_k=2),
        "700M": dict(d_model=896, n_layers=12, d_state=8, d_ff=1792, num_experts=8, moe_top_k=2),
        "1.5B": dict(d_model=1152, n_layers=16, d_state=8, d_ff=2304, num_experts=8, moe_top_k=2),
    }
    cfg_dict = configs.get(tier, configs["350M"])
    config = QUSSMConfig(
        vocab_size=50257,
        torch_dtype="float32",
        **cfg_dict
    )
    return config

class DistillationTrainer:
    def __init__(
        self,
        student_tier: str = "350M",
        teacher_model_id: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        learning_rate: float = 5e-4,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

        print(f"[*] Initializing Tokenizer ({teacher_model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[*] Loading Teacher Model ({teacher_model_id})...")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        print(f"[*] Initializing Scaled QU-SSM-MoE Student ({student_tier})...")
        self.student_config = create_scaled_config(student_tier)
        self.student = QUSSMForCausalLM(self.student_config).to(self.device)

        total_params = sum(p.numel() for p in self.student.parameters())
        print(f"[✓] Student Initialized: {total_params / 1e6:.2f}M Total Parameters")

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

    def train_step(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        
        # 1. Forward Teacher (No Gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            teacher_logits = teacher_outputs.logits

        # 2. Forward Student
        student_outputs = self.student(input_ids)
        student_logits = student_outputs.logits

        # Align vocab dimensions if needed
        vocab_min = min(student_logits.size(-1), teacher_logits.size(-1))
        s_logits = student_logits[..., :vocab_min]
        t_logits = teacher_logits[..., :vocab_min]

        # Shift logits for next-token prediction
        shift_s_logits = s_logits[..., :-1, :].contiguous().view(-1, vocab_min)
        shift_t_logits = t_logits[..., :-1, :].contiguous().view(-1, vocab_min)
        shift_labels = input_ids[..., 1:].contiguous().view(-1)

        # 3. Hard Loss (Standard Cross Entropy on ground truth)
        ce_loss = F.cross_entropy(shift_s_logits, shift_labels, ignore_index=-100)

        # 4. Soft Loss (KL Divergence with Teacher probability distribution)
        p_student = F.log_softmax(shift_s_logits / self.temperature, dim=-1)
        p_teacher = F.softmax(shift_t_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (self.temperature ** 2)

        # Combined Loss
        total_loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item(), ce_loss.item(), kl_loss.item()

    def generate_sample(self, prompt: str, max_new_tokens: int = 50) -> str:
        self.student.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            output_ids = self.student.generate(input_ids, max_new_tokens=max_new_tokens)
        self.student.train()
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("=== SCALED QU-SSM-MoE DISTILLATION TRAINER ===")
    trainer = DistillationTrainer(
        student_tier="350M",
        teacher_model_id="HuggingFaceTB/SmolLM-135M-Instruct",
        learning_rate=5e-4
    )
    print("\nSample Untrained Generation:")
    print(trainer.generate_sample("Once upon a time in a futuristic world,"))
