from dataclasses import dataclass
from typing import List

from src.util.hparams import HyperParams


@dataclass
class BCFHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    v_prob_threshold: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: int
    context_template_length_params: List[List[int]]

    # Module templates for MLP layers
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Module templates for cross-attention layers
    cross_attention_layers: List[int]
    cross_attention_module_tmp: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # Other Utils
    stats_dir: str