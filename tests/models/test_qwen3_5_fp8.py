from llm_mini_infer.models.fp8 import QuantConfig
from llm_mini_infer.models.qwen3_5.model import Qwen35Config, map_weight_name
from tests.support.tiny_gemma import REDHAT_FP8_BLOCK


def _cfg(quant: QuantConfig | None = None) -> Qwen35Config:
    return Qwen35Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        layer_types=("full_attention",),
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        rotary_dim=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        lin_k_heads=4,
        lin_v_heads=4,
        lin_k_dim=32,
        lin_v_dim=32,
        conv_kernel=4,
        tie_word_embeddings=True,
        max_position_embeddings=128,
        mtp_layers=1,
        quant=quant,
    )


def test_map_weight_name_default_scale_suffix():
    for quant in (None, QuantConfig.from_hf({"quant_method": "fp8", "weight_block_size": [128, 128]})):
        cfg = _cfg(quant)
        assert map_weight_name(
            cfg, "model.layers.0.mlp.gate_proj.weight_scale_inv"
        ) == ("layers.0.mlp.gate_up_proj.weight_scale", 0)
        assert map_weight_name(
            cfg, "model.layers.0.mlp.up_proj.weight_scale_inv"
        ) == ("layers.0.mlp.gate_up_proj.weight_scale", 2)
        assert map_weight_name(
            cfg, "model.layers.0.mlp.down_proj.weight_scale_inv"
        ) == ("layers.0.mlp.down_proj.weight_scale", 0)
           assert map_weight_name(cfg, "model.layers.0.self_attn.k_proj.weight_scale_inv") == ("layers.0.self_attn.qkv_proj.weight_scale", (cfg.num_heads * cfg.head_dim * (2 if cfg.attn_output_gate else 1)) // 128)
        assert map_weight_name(cfg, "model.layers.0.linear_attn.in_proj_z.weight_scale_inv") == ("layers.0.linear_attn.in_proj_qkvz.weight_scale", cfg.conv_dim // 128)
        assert map_weight_name(
            cfg, "mtp.layers.0.mlp.gate_proj.weight_scale_inv", mtp=True
        ) == ("layers.0.mlp.gate_up_proj.weight_scale", 0)
        assert map_weight_name(
            cfg, "mtp.layers.0.mlp.up_proj.weight_scale_inv", mtp=True
        ) == ("layers.0.mlp.gate_up_proj.weight_scale", 2)
     

def test_map_weight_name_compressed_tensors_scale_suffix():
    cfg = _cfg(
        QuantConfig.from_hf(REDHAT_FP8_BLOCK)
    )
    assert map_weight_name(
        cfg, "model.layers.0.mlp.gate_proj.weight_scale"
    ) == ("layers.0.mlp.gate_up_proj.weight_scale", 0)
    assert map_weight_name(
        cfg, "model.layers.0.mlp.up_proj.weight_scale"
    ) == ("layers.0.mlp.gate_up_proj.weight_scale", 2)
    assert map_weight_name(
        cfg, "model.layers.0.mlp.down_proj.weight_scale"
    ) == ("layers.0.mlp.down_proj.weight_scale", 0)
    assert map_weight_name(cfg, "model.layers.0.self_attn.k_proj.weight_scale") == ("layers.0.self_attn.qkv_proj.weight_scale", (cfg.num_heads * cfg.head_dim * (2 if cfg.attn_output_gate else 1)) // 128)
      assert map_weight_name(cfg, "model.layers.0.linear_attn.in_proj_z.weight_scale") == ("layers.0.linear_attn.in_proj_qkvz.weight_scale", cfg.conv_dim // 128)
    assert map_weight_name(
        cfg, "mtp.layers.0.mlp.gate_proj.weight_scale", mtp=True
    ) == ("layers.0.mlp.gate_up_proj.weight_scale", 0)
    assert map_weight_name(
        cfg, "mtp.layers.0.mlp.up_proj.weight_scale", mtp=True
    ) == ("layers.0.mlp.gate_up_proj.weight_scale", 2)
