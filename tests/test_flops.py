"""The FLOPs counter reproduces the Mixed-GGNAS tables (thop 0.1.1 convention, per training batch)."""

import pytest

from qenas.config import build_config
from qenas.models.network import build_model
from qenas.models.baselines import AttentionUNet, UNet
from qenas.models.summary import model_summary, thop_flops


def test_unet_reproduces_mixed_ggnas_table_values():
    # The Mixed-GGNAS U-Net has a two-class softmax output: Table 4/5 (256 x 256, batch 8) 437.94 G,
    # Table 7 (IDRiD 512 x 320, batch 4) 547.42 G - reproduced exactly.
    m = UNet(num_classes=2)
    assert sum(p.numel() for p in m.parameters()) / 1e6 == pytest.approx(31.04, abs=0.01)
    assert thop_flops(m, [8, 3, 256, 256]) / 1e9 == pytest.approx(437.94, abs=0.01)
    assert thop_flops(m, [4, 3, 320, 512]) / 1e9 == pytest.approx(547.42, abs=0.03)
    # The study's baseline has one logit per pixel (the QENAS loss): the head is 64 parameters smaller.
    assert thop_flops(UNet(), [8, 3, 256, 256]) / 1e9 == pytest.approx(437.94, abs=0.1)


def test_attention_unet_is_the_mixed_ggnas_variant():
    m = AttentionUNet()
    assert sum(p.numel() for p in m.parameters()) / 1e6 == pytest.approx(8.73, abs=0.01)
    # Table 5 (BUSI, 256 x 256, batch 8): 134.25 G; Table 7 (IDRiD, batch 4): 168.19 G  -> within 0.5 %
    assert thop_flops(m, [8, 3, 256, 256]) / 1e9 == pytest.approx(134.25, rel=5e-3)
    assert thop_flops(m, [4, 3, 320, 512]) / 1e9 == pytest.approx(168.19, rel=5e-3)


def test_model_summary_flops_scale_with_batch():
    cfg = build_config("BUSI", overrides={"data.image_size": [64, 64], "model.base_channels": 16})
    m = build_model([0, 1, 2, 3, 4, 3, 2, 1], cfg)
    s = model_summary(m, [64, 64], 3, 8)
    assert s["gflops_mixed_ggnas"] == pytest.approx(8 * s["gflops_per_image"])
    assert s["total_params"] == sum(p.numel() for p in m.parameters())
