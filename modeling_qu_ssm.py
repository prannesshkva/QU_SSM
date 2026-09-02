import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers import PreTrainedModel, GenerationMixin, AutoConfig
from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput

class QUSSMConfig(PretrainedConfig):
    model_type = "qu_ssm_moe"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_layers: int = 6,
        d_state: int = 8,
        d_ff: int = 1024,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.moe_top_k = min(moe_top_k, num_experts)
        super().__init__(**kwargs)

    def __getattr__(self, key: str):
        if key.startswith("use_") or key.startswith("is_"):
            return False
        if key in ("all_tied_weights_keys", "_tied_weights_keys"):
            return []
        raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}")

try:
    AutoConfig.register("qu_ssm_moe", QUSSMConfig)
except Exception:
    pass

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
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=True)
        self.gamma_proj = nn.Linear(d_model, d_model, bias=True)
        self.theta_proj = nn.Linear(d_model, d_model, bias=False)
        self.theta_base = nn.Parameter(torch.linspace(0.01, 0.5, d_state).view(1, 1, 1, d_state).repeat(1, 1, d_model, 1))
        self.C_proj = nn.Linear(d_model, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, L, D = x.shape
        N = self.d_state
        x_norm = self.norm(x)
        u_val, gate = self.in_proj(x_norm).chunk(2, dim=-1)
        u = u_val.unsqueeze(-1).expand(B, L, D, N)
        theta = (self.theta_proj(x_norm).unsqueeze(-1) + self.theta_base)
        log_g = F.logsigmoid(self.gamma_proj(x_norm)).unsqueeze(-1).expand(B, L, D, N)
        
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
        h_sum = h_exact.sum(dim=-1)
        y_ssm = self.C_proj(h_sum) + x * self.D
        return y_ssm * F.silu(gate)

class QUSSMBlock(nn.Module):
    def __init__(self, config: QUSSMConfig):
        super().__init__()
        self.ssm = ExactRealQUBlock(d_model=config.d_model, d_state=config.d_state)
        self.mlp_norm = RMSNorm(config.d_model)
        self.moe = StaticTPUMoE(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_experts=config.num_experts,
            moe_top_k=config.moe_top_k
        ) if config.num_experts > 1 else SwiGLUExpert(config.d_model, config.d_ff)

    def forward(self, x):
        x = x + self.ssm(x)
        x = x + self.moe(self.mlp_norm(x))
        return x

class QUSSMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = QUSSMConfig
    _tied_weights_keys = ["lm_head.weight"]
    all_tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["QUSSMBlock"]

    def __init__(self, config: QUSSMConfig):
        super().__init__(config)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([QUSSMBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

class QUSSMForAudio(PreTrainedModel):
    config_class = QUSSMConfig
    _tied_weights_keys = []
    all_tied_weights_keys = []

    def __init__(self, config: QUSSMConfig, num_classes: int = 10, patch_size: int = 16):
        super().__init__(config)
        self.config = config
        self.patch_embed = nn.Conv1d(1, config.d_model, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList([QUSSMBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(self, input_values, labels=None):
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        x = self.patch_embed(input_values).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        h = self.final_norm(x)
        logits = self.classifier(h.mean(dim=1))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

class QUSSMForSensorTelemetry(PreTrainedModel):
    config_class = QUSSMConfig
    _tied_weights_keys = []
    all_tied_weights_keys = []

    def __init__(self, config: QUSSMConfig, input_dim: int = 1, output_dim: int = 1):
        super().__init__(config)
        self.config = config
        self.in_proj = nn.Linear(input_dim, config.d_model)
        self.layers = nn.ModuleList([QUSSMBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, output_dim)

    def forward(self, input_telemetry, labels=None):
        x = self.in_proj(input_telemetry)
        for layer in self.layers:
            x = layer(x)
        h = self.final_norm(x)
        predictions = self.out_proj(h)
        loss = None
        if labels is not None:
            loss = F.mse_loss(predictions, labels)
        return {"loss": loss, "predictions": predictions}

class VisionQUSSM(PreTrainedModel):
    config_class = QUSSMConfig
    _tied_weights_keys = []
    all_tied_weights_keys = []

    def __init__(self, config: QUSSMConfig, img_size: int = 224, patch_size: int = 16, num_classes: int = 1000):
        super().__init__(config)
        self.config = config
        self.patch_embed = nn.Conv2d(3, config.d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.layers = nn.ModuleList([QUSSMBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(self, pixel_values, labels=None):
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        h = self.final_norm(x)
        logits = self.classifier(h.mean(dim=1))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
