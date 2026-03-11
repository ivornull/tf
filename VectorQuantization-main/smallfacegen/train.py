"""
train.py  —  两阶段训练：VQVAE+GAN → GPT Transformer 先验

Stage 1：VQVAE（Encoder + VQ + Decoder）照常训练，
         额外引入 PatchGAN Discriminator 提升重构锐度。
         VQVAE 本身结构不变，判别器只在 train.py 里参与训练。

Stage 2：冻结 VQVAE，用 GPT Decoder-only Transformer
         学习离散码本索引的先验，实现无条件图像生成。

Usage:
    python train.py                        # 两阶段都跑
    python train.py --stage 1              # 只跑 VQVAE+GAN
    python train.py --stage 2              # 只跑 Transformer（需要已有 VQVAE checkpoint）
    python train.py --stage 2 --vqvae_ckpt outputs/checkpoints/vqvae_epoch_50.pth
"""

import os

# ── 路径 & 训练配置（直接在这里改，不需要命令行参数）────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

STAGE       = 0        # 0=两阶段都跑  1=只跑VQVAE  2=只跑Transformer
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
VQVAE_CKPT  = None     # STAGE=2 时填路径，例："outputs/checkpoints/vqvae_epoch_50.pth"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import yaml
from torch.utils.data import DataLoader

from models.VQVAE import VQVAE
from models.Transformer import GPTDecoder
from models.Discrimination import PatchGANDiscriminator, hinge_d_loss, hinge_g_loss, adopt_weight
from utils.data_loader import get_animal_dataset
from utils.logger import Logger


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(state_dict, path):
    torch.save(state_dict, path)
    print(f"  Saved → {path}")


# ── Stage 1: VQVAE + PatchGAN Discriminator ───────────────────────────────────

def train_vqvae(config, device, logger, dataloader):
    cfg_m   = config["model"]
    cfg_t   = config["train"]
    cfg_d   = config["discriminator"]
    cfg_log = config["log"]

    # ── 初始化 VQVAE（原有结构完全不动）──────────────────────────────────────
    vqvae = VQVAE(
        in_channels     = cfg_m["in_channels"],
        out_channels    = cfg_m["out_channels"],
        hidden_channels = cfg_m["hidden_channels"],
        res_nums        = cfg_m["residual_blocks"],
        n_e             = cfg_m["n_e"],
        e_dim           = cfg_m["e_dim"],
        beta            = cfg_m["beta"],
        device          = device,
    ).to(device)

    # ── 初始化判别器（独立模块，不塞进 VQVAE）────────────────────────────────
    discriminator = PatchGANDiscriminator(
        in_channels   = cfg_m["in_channels"],
        base_channels = cfg_d["base_channels"],
        n_layers      = cfg_d["n_layers"],
    ).to(device)

    # ── 两个独立优化器 ────────────────────────────────────────────────────────
    vqvae_opt = optim.Adam(
        vqvae.parameters(),
        lr=float(cfg_t["lr"]),
        betas=(0.5, 0.9),
    )
    disc_opt = optim.Adam(
        discriminator.parameters(),
        lr=float(cfg_t.get("disc_lr", cfg_t["lr"])),
        betas=(0.5, 0.9),
    )

    disc_weight = cfg_t["disc_weight"]
    disc_start  = cfg_t["disc_start"]
    num_epochs  = cfg_t["epochs"]
    save_every  = cfg_t["save_every"]
    log_every   = cfg_log["log_interval"]
    global_step = 0

    print("\n" + "=" * 60)
    print("  Stage 1: VQVAE + PatchGAN Discriminator Training")
    print("=" * 60)

    for epoch in range(num_epochs):
        vqvae.train()
        discriminator.train()
        epoch_total = 0.0

        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            # ── VQVAE 前向（total_loss = recon_loss + vq_loss，原始逻辑）────
            total_loss, recon, recon_loss, vq_loss, perplexity, quantized, indices = vqvae(batch)

            # ── 判断是否启用 GAN 损失 ─────────────────────────────────────────
            g_weight = adopt_weight(disc_weight, global_step, disc_start)

            # ── Generator 步：VQVAE 损失 + 对抗损失 ──────────────────────────
            if g_weight > 0:
                fake_logits = discriminator(recon)
                adv_loss    = hinge_g_loss(fake_logits)
            else:
                adv_loss = torch.tensor(0.0, device=device)

            vqvae_total = total_loss + g_weight * adv_loss

            vqvae_opt.zero_grad()
            vqvae_total.backward()
            vqvae_opt.step()

            # ── Discriminator 步（warm-up 期间跳过）──────────────────────────
            d_loss_val = 0.0
            if global_step >= disc_start:
                # 重新前向但 detach，防止梯度流回 VQVAE
                with torch.no_grad():
                    _, recon_d, *_ = vqvae(batch)

                real_logits = discriminator(batch.detach())
                fake_logits = discriminator(recon_d.detach())
                d_loss      = hinge_d_loss(real_logits, fake_logits)

                disc_opt.zero_grad()
                d_loss.backward()
                disc_opt.step()
                d_loss_val = d_loss.item()

            epoch_total += vqvae_total.item()
            global_step += 1

            # ── 日志 ──────────────────────────────────────────────────────────
            if global_step % log_every == 0:
                print(
                    f"[VQVAE] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {i+1}/{len(dataloader)}  "
                    f"total={vqvae_total.item():.4f}  "
                    f"recon={recon_loss.item():.4f}  "
                    f"vq={vq_loss.item():.4f}  "
                    f"adv(G)={adv_loss.item():.4f}  "
                    f"disc(D)={d_loss_val:.4f}  "
                    f"perp={perplexity.item():.1f}  "
                    f"g_w={g_weight:.2f}"
                )
                logger.log_scalar("vqvae/total",  vqvae_total.item(),  global_step)
                logger.log_scalar("vqvae/recon",  recon_loss.item(),   global_step)
                logger.log_scalar("vqvae/vq",     vq_loss.item(),      global_step)
                logger.log_scalar("vqvae/adv_g",  adv_loss.item(),     global_step)
                logger.log_scalar("vqvae/disc_d", d_loss_val,          global_step)
                logger.log_scalar("vqvae/perp",   perplexity.item(),   global_step)

        # ── 每隔 save_every 个 epoch 保存样本 + 模型 ─────────────────────────
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            vqvae.eval()
            with torch.no_grad():
                sample    = batch[:8]
                recon_vis = vqvae.generate(sample)
                grid      = torch.cat([sample, recon_vis], dim=0)
                vutils.save_image(
                    grid,
                    os.path.join(BASE_DIR, "outputs", "samples", f"vqvae_recon_epoch_{epoch+1}.png"),
                    nrow=8, normalize=True,
                )
            save_checkpoint(vqvae.state_dict(),
                            os.path.join(BASE_DIR, "outputs", "checkpoints", f"vqvae_epoch_{epoch+1}.pth"))
            save_checkpoint(discriminator.state_dict(),
                            os.path.join(BASE_DIR, "outputs", "checkpoints", f"disc_epoch_{epoch+1}.pth"))

        print(f"  → Epoch {epoch+1} avg loss: {epoch_total / len(dataloader):.4f}\n")

    return vqvae


# ── Stage 2: GPT Transformer 先验 ─────────────────────────────────────────────

def train_transformer(config, device, logger, dataloader, vqvae):
    cfg_m   = config["model"]
    cfg_tr  = config["transformer"]
    cfg_log = config["log"]
    img_sz  = config["data"]["image_size"]

    # Encoder 下采样 4×（两次 stride=2 卷积）
    lat_size = img_sz // 4
    seq_len  = lat_size * lat_size

    transformer = GPTDecoder(
        vocab_size = cfg_m["n_e"],
        seq_len    = seq_len,
        n_embd     = cfg_tr["n_embd"],
        n_head     = cfg_tr["n_head"],
        n_layer    = cfg_tr["n_layer"],
        dropout    = cfg_tr["dropout"],
    ).to(device)

    optimizer    = optim.AdamW(transformer.parameters(), lr=float(cfg_tr["lr"]))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_tr["epochs"] * len(dataloader)
    )
    criterion = nn.CrossEntropyLoss()

    num_epochs  = cfg_tr["epochs"]
    log_every   = cfg_log["log_interval"]
    global_step = 0

    # 冻结 VQVAE，只用来提取离散索引
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    print("\n" + "=" * 60)
    print("  Stage 2: GPT Transformer Prior Training")
    print(f"  latent grid: {lat_size}×{lat_size}  →  seq_len = {seq_len}")
    print("=" * 60)

    for epoch in range(num_epochs):
        transformer.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            # 用冻结的 VQVAE 获取离散索引
            with torch.no_grad():
                _, _, _, _, _, _, indices = vqvae(batch)   # (B, lat, lat)
            indices = indices.view(batch.size(0), -1)       # (B, seq_len)

            # Teacher-forcing：输入 t=0..T-2，预测 t=1..T-1
            inp    = indices[:, :-1]
            target = indices[:, 1:]

            logits = transformer(inp)                       # (B, T-1, vocab_size)
            loss   = criterion(logits.permute(0, 2, 1), target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Transformer] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {i+1}/{len(dataloader)}  "
                    f"loss={loss.item():.4f}  lr={lr:.6f}"
                )
                logger.log_scalar("transformer/loss", loss.item(), global_step)

        print(f"  → Epoch {epoch+1} avg loss: {epoch_loss / len(dataloader):.4f}")
        save_checkpoint(transformer.state_dict(),
                        os.path.join(BASE_DIR, "outputs", "checkpoints", f"transformer_epoch_{epoch+1}.pth"))

    return transformer


# ── 推理：Transformer 采样 → VQVAE 解码 → 新图像 ──────────────────────────────

@torch.no_grad()
def generate_samples(config, device, vqvae, transformer,
                     n_samples=16,
                     out_path=os.path.join(BASE_DIR, "outputs", "samples", "generated_final.png")):
    cfg_tr  = config["transformer"]
    img_sz  = config["data"]["image_size"]
    lat_sz  = img_sz // 4
    seq_len = lat_sz * lat_sz

    vqvae.eval()
    transformer.eval()

    print(f"\n生成 {n_samples} 张新图像 …")

    prompt  = torch.zeros(n_samples, 1, dtype=torch.long, device=device)
    indices = transformer.generate(
        prompt,
        n_new_tokens = seq_len - 1,
        temperature  = cfg_tr.get("temperature", 1.0),
        top_k        = cfg_tr.get("top_k"),
    )                                                  # (n_samples, seq_len)

    # 离散索引 → 码本向量 → VQVAE 解码
    indices          = indices.view(n_samples, lat_sz, lat_sz)
    codebook_weights = vqvae.vq.embedding.weight       # (n_e, e_dim)
    z_q              = codebook_weights[indices]        # (B, H, W, e_dim)
    z_q              = z_q.permute(0, 3, 1, 2).contiguous()
    generated        = vqvae.decoder(z_q)              # (B, C, H, W)

    vutils.save_image(generated, out_path, nrow=4, normalize=True)
    print(f"  已保存 → {out_path}")


def main():
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    make_dirs(
        os.path.join(BASE_DIR, "outputs", "samples"),
        os.path.join(BASE_DIR, "outputs", "checkpoints"),
        config["log"]["log_dir"],
    )
    logger = Logger(config)

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    img_sz    = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize(img_sz),
        transforms.CenterCrop(img_sz),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ])
    dataset = get_animal_dataset(
        root      = config["data"]["dataset_path"],
        transform = transform,
        animal    = config["data"].get("animal", "dog"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = config["train"]["batch_size"],
        shuffle     = True,
        num_workers = 0,   # Windows 下建议设 0 避免多进程死锁
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )

    # ── Stage 1：训练 VQVAE + Discriminator ───────────────────────────────────
    if STAGE in (0, 1):
        vqvae = train_vqvae(config, device, logger, dataloader)

    # ── Stage 2：训练 Transformer ─────────────────────────────────────────────
    if STAGE in (0, 2):
        if STAGE == 2:
            # 只跑 Stage 2 时从 checkpoint 加载 VQVAE
            cfg_m = config["model"]
            vqvae = VQVAE(
                in_channels     = cfg_m["in_channels"],
                out_channels    = cfg_m["out_channels"],
                hidden_channels = cfg_m["hidden_channels"],
                res_nums        = cfg_m["residual_blocks"],
                n_e             = cfg_m["n_e"],
                e_dim           = cfg_m["e_dim"],
                beta            = cfg_m["beta"],
                device          = device,
            ).to(device)

            ckpt = VQVAE_CKPT
            if ckpt is None:
                # 自动找 outputs/checkpoints/ 下最新的 vqvae_epoch_*.pth
                ckpt_dir = os.path.join(BASE_DIR, "outputs", "checkpoints")
                ckpts = sorted(
                    [f for f in os.listdir(ckpt_dir)
                     if f.startswith("vqvae") and f.endswith(".pth")],
                    key=lambda x: int(x.replace("vqvae_epoch_", "").replace(".pth", ""))
                )
                if not ckpts:
                    raise FileNotFoundError(
                        "找不到 VQVAE checkpoint，请先把 STAGE 设为 0 或 1 跑一次。"
                    )
                ckpt = os.path.join(ckpt_dir, ckpts[-1])

            print(f"加载 VQVAE checkpoint: {ckpt}")
            vqvae.load_state_dict(torch.load(ckpt, map_location=device))

        transformer = train_transformer(config, device, logger, dataloader, vqvae)
        generate_samples(config, device, vqvae, transformer)

    print("\n✓ 训练完成。")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()