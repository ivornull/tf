"""
test_pipeline.py — 用一张随机假图片跑通完整流程，不需要真实数据集

测试覆盖：
  1. Encoder 前向
  2. VectorQuantizer 前向
  3. Decoder 前向
  4. VQVAE 完整前向 + generate()
  5. PatchGAN Discriminator 前向（real/fake）
  6. 判别器 hinge loss 计算
  7. Generator 对抗 loss 计算
  8. GPT Transformer 前向
  9. Transformer 自回归 generate()
  10. 推理流程：Transformer 采样索引 → 码本查表 → Decoder 生成图像

运行方式（在项目根目录 smallfacegen/ 下执行）：
    python test_pipeline.py
"""

import sys
import os
import torch
import torch.nn.functional as F

# ── 让 Python 找到 models/ utils/ ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.Encoder import Encoder
from models.Decoder import Decoder
from models.VectorQuantization import VectorQuantizer
from models.VQVAE import VQVAE
from models.Discrimination import (
    PatchGANDiscriminator, hinge_d_loss, hinge_g_loss, adopt_weight
)
from models.Transformer import GPTDecoder


# ── 测试配置（与 config.yaml 保持一致）────────────────────────────────────────
IN_CH      = 3
OUT_CH     = 64      # e_dim
HIDDEN_CH  = 128
RES_NUMS   = 2
N_E        = 512     # 码本大小
E_DIM      = 64
BETA       = 0.25
IMG_SIZE   = 128     # 测试用小尺寸
BATCH      = 1       # 只用 1 张图

LAT_SIZE   = IMG_SIZE // 4          # 32
SEQ_LEN    = LAT_SIZE * LAT_SIZE    # 1024

DEVICE = torch.device("cpu")        # 测试用 CPU，避免 GPU 依赖

# ── 颜色打印 ──────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

passed = []
failed = []


def ok(name):
    print(f"  {GREEN}✓ PASS{RESET}  {name}")
    passed.append(name)


def fail(name, e):
    print(f"  {RED}✗ FAIL{RESET}  {name}")
    print(f"         {RED}{type(e).__name__}: {e}{RESET}")
    failed.append(name)


def section(title):
    print(f"\n{YELLOW}{'─'*55}{RESET}")
    print(f"{YELLOW}  {title}{RESET}")
    print(f"{YELLOW}{'─'*55}{RESET}")


# ── 假图片（1 张 RGB 128×128）────────────────────────────────────────────────
fake_image = torch.randn(BATCH, IN_CH, IMG_SIZE, IMG_SIZE)


# ═══════════════════════════════════════════════════════
#  1. Encoder
# ═══════════════════════════════════════════════════════
section("1. Encoder")
try:
    encoder = Encoder(IN_CH, OUT_CH, HIDDEN_CH, RES_NUMS)
    z = encoder(fake_image)
    assert z.shape == (BATCH, OUT_CH, LAT_SIZE, LAT_SIZE), \
        f"Expected ({BATCH},{OUT_CH},{LAT_SIZE},{LAT_SIZE}), got {z.shape}"
    ok(f"Encoder output shape: {tuple(z.shape)}")
except Exception as e:
    fail("Encoder", e)
    z = torch.randn(BATCH, OUT_CH, LAT_SIZE, LAT_SIZE)  # fallback


# ═══════════════════════════════════════════════════════
#  2. VectorQuantizer
# ═══════════════════════════════════════════════════════
section("2. VectorQuantizer")
try:
    vq = VectorQuantizer(N_E, E_DIM, BETA, DEVICE)
    vq_loss, z_q, perplexity, min_encodings, indices = vq(z)

    assert z_q.shape == z.shape,         f"z_q shape mismatch: {z_q.shape}"
    assert indices.shape == (BATCH, LAT_SIZE, LAT_SIZE), \
        f"indices shape: {indices.shape}"
    assert vq_loss.ndim == 0,            "vq_loss should be scalar"
    assert perplexity.ndim == 0,         "perplexity should be scalar"

    ok(f"z_q shape:          {tuple(z_q.shape)}")
    ok(f"indices shape:      {tuple(indices.shape)}")
    ok(f"vq_loss:            {vq_loss.item():.4f}")
    ok(f"perplexity:         {perplexity.item():.2f}")
except Exception as e:
    fail("VectorQuantizer", e)
    z_q     = z.detach()
    indices = torch.zeros(BATCH, LAT_SIZE, LAT_SIZE, dtype=torch.long)


# ═══════════════════════════════════════════════════════
#  3. Decoder
# ═══════════════════════════════════════════════════════
section("3. Decoder")
try:
    decoder = Decoder(IN_CH, OUT_CH, HIDDEN_CH, RES_NUMS)
    recon   = decoder(z_q)
    assert recon.shape == fake_image.shape, \
        f"Expected {fake_image.shape}, got {recon.shape}"
    ok(f"Decoder output shape: {tuple(recon.shape)}")
except Exception as e:
    fail("Decoder", e)
    recon = fake_image.clone()


# ═══════════════════════════════════════════════════════
#  4. VQVAE 完整前向 + generate()
# ═══════════════════════════════════════════════════════
section("4. VQVAE forward + generate()")
try:
    vqvae = VQVAE(IN_CH, OUT_CH, HIDDEN_CH, RES_NUMS, N_E, E_DIM, BETA, DEVICE)

    total_loss, recon2, recon_loss, vq_loss2, perplexity2, quantized, indices2 = vqvae(fake_image)

    assert recon2.shape == fake_image.shape,  f"recon shape: {recon2.shape}"
    assert quantized.shape == (BATCH, OUT_CH, LAT_SIZE, LAT_SIZE)
    assert indices2.shape  == (BATCH, LAT_SIZE, LAT_SIZE)
    assert total_loss.ndim == 0

    ok(f"VQVAE forward — total_loss: {total_loss.item():.4f}  "
       f"recon: {recon_loss.item():.4f}  vq: {vq_loss2.item():.4f}")

    gen = vqvae.generate(fake_image)
    assert gen.shape == fake_image.shape
    ok(f"VQVAE.generate() shape: {tuple(gen.shape)}")

    # 反向传播测试
    total_loss.backward()
    ok("VQVAE backward pass OK")
except Exception as e:
    fail("VQVAE", e)


# ═══════════════════════════════════════════════════════
#  5. PatchGAN Discriminator 前向
# ═══════════════════════════════════════════════════════
section("5. PatchGAN Discriminator forward")
try:
    disc = PatchGANDiscriminator(in_channels=IN_CH, base_channels=64, n_layers=3)

    real_logits = disc(fake_image)
    fake_logits = disc(recon2.detach())

    assert real_logits.shape[1] == 1,  f"output channel should be 1, got {real_logits.shape}"
    assert real_logits.shape == fake_logits.shape

    ok(f"Discriminator output shape: {tuple(real_logits.shape)}")
except Exception as e:
    fail("PatchGAN Discriminator forward", e)
    real_logits = torch.zeros(BATCH, 1, 14, 14)
    fake_logits = torch.zeros(BATCH, 1, 14, 14)


# ═══════════════════════════════════════════════════════
#  6. Discriminator hinge loss
# ═══════════════════════════════════════════════════════
section("6. Hinge loss (Discriminator + Generator)")
try:
    d_loss = hinge_d_loss(real_logits, fake_logits)
    g_loss = hinge_g_loss(fake_logits)
    assert d_loss.ndim == 0
    assert g_loss.ndim == 0
    ok(f"D hinge loss: {d_loss.item():.4f}")
    ok(f"G hinge loss: {g_loss.item():.4f}")

    w_before = adopt_weight(0.5, current_step=0,    disc_start=100)
    w_after  = adopt_weight(0.5, current_step=200,  disc_start=100)
    assert w_before == 0.0
    assert w_after  == 0.5
    ok(f"adopt_weight: step=0→{w_before}  step=200→{w_after}")
except Exception as e:
    fail("Hinge loss", e)


# ═══════════════════════════════════════════════════════
#  7. Stage 1 联合损失模拟（VQVAE + Discriminator）
# ═══════════════════════════════════════════════════════
section("7. Stage 1 联合损失模拟")
try:
    vqvae2 = VQVAE(IN_CH, OUT_CH, HIDDEN_CH, RES_NUMS, N_E, E_DIM, BETA, DEVICE)
    disc2  = PatchGANDiscriminator(IN_CH, 64, 3)

    total_loss3, recon3, recon_loss3, vq_loss3, perp3, _, _ = vqvae2(fake_image)

    g_weight = adopt_weight(0.5, current_step=200, disc_start=100)
    adv_loss = hinge_g_loss(disc2(recon3))
    vqvae_total = total_loss3 + g_weight * adv_loss

    vqvae_total.backward()   # VQVAE + G loss 反向
    ok(f"VQVAE generator total loss: {vqvae_total.item():.4f}  (adv weight={g_weight})")

    # Discriminator 单独反向
    real_l = disc2(fake_image.detach())
    fake_l = disc2(recon3.detach())
    d_loss2 = hinge_d_loss(real_l, fake_l)
    d_loss2.backward()
    ok(f"Discriminator loss backward OK  d_loss={d_loss2.item():.4f}")
except Exception as e:
    fail("Stage 1 joint loss", e)


# ═══════════════════════════════════════════════════════
#  8. GPT Transformer 前向
# ═══════════════════════════════════════════════════════
section("8. GPT Transformer forward")
try:
    transformer = GPTDecoder(
        vocab_size = N_E,
        seq_len    = SEQ_LEN,
        n_embd     = 64,    # 测试用小尺寸
        n_head     = 4,
        n_layer    = 2,
        dropout    = 0.0,
    )

    # 模拟一批索引序列
    fake_indices = torch.randint(0, N_E, (BATCH, SEQ_LEN))  # (1, 1024)

    inp    = fake_indices[:, :-1]   # (1, 1023)
    target = fake_indices[:, 1:]    # (1, 1023)

    logits = transformer(inp)
    assert logits.shape == (BATCH, SEQ_LEN - 1, N_E), \
        f"logits shape: {logits.shape}"

    loss = torch.nn.CrossEntropyLoss()(logits.permute(0, 2, 1), target)
    loss.backward()

    ok(f"Transformer logits shape: {tuple(logits.shape)}")
    ok(f"Transformer CE loss: {loss.item():.4f}")
    ok("Transformer backward pass OK")
except Exception as e:
    fail("GPT Transformer forward", e)


# ═══════════════════════════════════════════════════════
#  9. Transformer 自回归 generate()
# ═══════════════════════════════════════════════════════
section("9. Transformer autoregressive generate()")
try:
    prompt    = torch.zeros(BATCH, 1, dtype=torch.long)
    generated = transformer.generate(
        prompt,
        n_new_tokens = SEQ_LEN - 1,
        temperature  = 1.0,
        top_k        = 50,
    )
    assert generated.shape == (BATCH, SEQ_LEN), \
        f"generated shape: {generated.shape}"
    assert generated.min() >= 0 and generated.max() < N_E, \
        f"index out of range: [{generated.min()}, {generated.max()}]"

    ok(f"Generated sequence shape: {tuple(generated.shape)}")
    ok(f"Index range: [{generated.min().item()}, {generated.max().item()}]  (valid 0~{N_E-1})")
except Exception as e:
    fail("Transformer generate()", e)


# ═══════════════════════════════════════════════════════
#  10. 完整推理流程：索引 → 码本 → Decoder → 图像
# ═══════════════════════════════════════════════════════
section("10. 推理流程：Transformer 索引 → VQVAE Decoder → 图像")
try:
    # 用 VQVAE 的码本权重做查表
    codebook_weights = vqvae.vq.embedding.weight    # (N_E, E_DIM)
    idx_2d           = generated.view(BATCH, LAT_SIZE, LAT_SIZE)
    z_q_gen          = codebook_weights[idx_2d]     # (B, H, W, E_DIM)
    z_q_gen          = z_q_gen.permute(0, 3, 1, 2).contiguous()  # (B, E_DIM, H, W)

    with torch.no_grad():
        img_gen = vqvae.decoder(z_q_gen)            # (B, C, H, W)

    assert img_gen.shape == fake_image.shape, \
        f"Generated image shape: {img_gen.shape}"

    ok(f"Generated image shape: {tuple(img_gen.shape)}")
    ok("完整推理流程 OK ✓")
except Exception as e:
    fail("Full inference pipeline", e)


# ═══════════════════════════════════════════════════════
#  汇总
# ═══════════════════════════════════════════════════════
total = len(passed) + len(failed)
print(f"\n{'='*55}")
print(f"  结果：{GREEN}{len(passed)} passed{RESET}  /  {RED}{len(failed)} failed{RESET}  /  {total} total")
print(f"{'='*55}")

if failed:
    print(f"\n{RED}以下测试失败，请检查对应模块：{RESET}")
    for f in failed:
        print(f"  · {f}")
    sys.exit(1)
else:
    print(f"\n{GREEN}所有测试通过！流程可以正常运行。{RESET}\n")