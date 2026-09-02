from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig


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
        # Standard aliases expected by transformers internals
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.intermediate_size = d_ff
        self.num_attention_heads = 1
        self.max_position_embeddings = 131072
        super().__init__(**kwargs)


try:
    AutoConfig.register("qu_ssm_moe", QUSSMConfig)
except Exception:
    pass
