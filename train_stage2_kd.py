import os
from argparse import ArgumentParser
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, log_txt_as_img
from diffbir.sampler import SpacedSampler
from diffbir.losses.haar_wavelet_kd_loss import HaarWaveletKDLoss


def move_tensor_batch_to_device(batch, device):
    """
    GoProKDDataset 回傳 dict，裡面有 tensor 也有 string/list。
    不使用 diffbir.utils.common.to()，避免它嘗試把路徑字串搬到 GPU。
    """
    for k in ["gt", "cond", "teacher"]:
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def main(args) -> None:
    # Setup accelerator
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device

    cfg = OmegaConf.load(args.config)

    # Setup experiment folder
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")
    else:
        exp_dir = cfg.train.exp_dir
        ckpt_dir = os.path.join(exp_dir, "checkpoints")

    # Create CLDM / IRControlNet model
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)

    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)

    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(
            torch.load(cfg.train.resume, map_location="cpu")
        )
        if accelerator.is_main_process:
            print(f"strictly load controlnet weight from checkpoint: {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                "strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # KD settings
    use_kd = cfg.train.get("use_kd", True)
    kd_weight = cfg.train.get("kd_weight", 0.5)

    use_wavelet_kd = cfg.train.get("use_wavelet_kd", False)
    wavelet_kd_weight = cfg.train.get("wavelet_kd_weight", 0.5)
    wavelet_high_freq_weight = cfg.train.get("wavelet_high_freq_weight", 2.0)

    # 為了省 VRAM，預設不做 image sampling log
    enable_image_log = cfg.train.get("enable_image_log", False)

    use_wandb = cfg.train.get("use_wandb", False)
    wandb_project = cfg.train.get("wandb_project", "DiffBIR-GoPro-KD")
    wandb_entity = cfg.train.get("wandb_entity", None)
    wandb_name = cfg.train.get("wandb_name", None)
    wandb_tags = cfg.train.get("wandb_tags", [])

    wavelet_kd_loss = HaarWaveletKDLoss(
        high_freq_weight=wavelet_high_freq_weight
    )

    
    # Freeze all parameters except controlnet / IRControlNet.
    # 這可以避免 VAE / SD UNet 產生不必要梯度，降低 VRAM。
    for p in cldm.parameters():
        p.requires_grad = False

    for p in cldm.controlnet.parameters():
        p.requires_grad = True


    # Optimizer: 只訓練 controlnet / IRControlNet
    opt = torch.optim.AdamW(
        cldm.controlnet.parameters(),
        lr=cfg.train.learning_rate,
    )

    # Dataset
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")
        print(f"use_kd: {use_kd}, kd_weight: {kd_weight}")
        print(
            f"use_wavelet_kd: {use_wavelet_kd}, "
            f"wavelet_kd_weight: {wavelet_kd_weight}, "
            f"wavelet_high_freq_weight: {wavelet_high_freq_weight}"
        )
        print(f"enable_image_log: {enable_image_log}")

    # Prepare models
    cldm.train().to(device)
    diffusion.to(device)
    wavelet_kd_loss.to(device)

    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Training variables
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0

    step_total_losses = []
    step_diff_losses = []
    step_out_kd_losses = []
    step_wav_kd_losses = []

    epoch_total_losses = []

    if enable_image_log:
        sampler = SpacedSampler(
            diffusion.betas,
            diffusion.parameterization,
            rescale_cfg=False,
        )
    else:
        sampler = None

    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)

        if use_wandb:
            if wandb is None:
                raise ImportError(
                    "use_wandb=True, but wandb is not installed. "
                    "Please run: pip install wandb"
                )

            wandb_config = OmegaConf.to_container(cfg, resolve=True)

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                tags=list(wandb_tags) if wandb_tags is not None else None,
                config=wandb_config,
                dir=exp_dir,
            )

            wandb.define_metric("global_step")
            wandb.define_metric("loss/*", step_metric="global_step")
            wandb.define_metric("train/*", step_metric="global_step")

        print(f"Training for {max_steps} steps...")
    else:
        writer = None

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )

        for batch in loader:
            batch = move_tensor_batch_to_device(batch, device)

            # Dataset 回傳 HWC tensor
            gt = batch["gt"]              # [-1, 1], BHWC
            clean = batch["cond"]         # [0, 1], BHWC, NAFNet-WKD condition
            teacher_img = batch["teacher"]  # [-1, 1], BHWC, offline DiffBIR teacher

            prompt = batch.get("prompt", "")
            if isinstance(prompt, str):
                prompt = [prompt] * gt.shape[0]

            # BHWC -> BCHW
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            clean = rearrange(clean, "b h w c -> b c h w").contiguous().float()
            teacher_img = rearrange(
                teacher_img, "b h w c -> b c h w"
            ).contiguous().float()

            with torch.no_grad():
                # GT sharp image latent
                z_0 = pure_cldm.vae_encode(gt)

                # 直接使用 NAFNet-WKD condition，不再使用 SwinIR
                cond = pure_cldm.prepare_condition(clean, prompt)

                # noise augmentation on condition latent
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0,
                            noise_aug_timestep,
                            (z_0.shape[0],),
                            device=device,
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )

            t = torch.randint(
                0,
                diffusion.num_timesteps,
                (z_0.shape[0],),
                device=device,
            )

            # 需要你已經在 gaussian_diffusion.py 新增 p_losses_with_x0()
            loss_diff, pred_x0 = diffusion.p_losses_with_x0(
                cldm,
                z_0,
                t,
                cond_aug,
            )

            # Decode predicted x0 latent to image space.
            # VAE params 不會被 optimizer 更新，但需要 gradient wrt pred_x0。
            student_img = pure_cldm.vae_decode(pred_x0)

            loss = loss_diff

            loss_out_kd = torch.tensor(0.0, device=device)
            loss_wav_kd = torch.tensor(0.0, device=device)

            if use_kd:
                loss_out_kd = F.l1_loss(student_img, teacher_img) * kd_weight
                loss = loss + loss_out_kd

            if use_wavelet_kd:
                loss_wav_kd = (
                    wavelet_kd_loss(student_img, teacher_img)
                    * wavelet_kd_weight
                )
                loss = loss + loss_wav_kd

            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            opt.step()
            accelerator.wait_for_everyone()

            global_step += 1

            step_total_losses.append(loss.detach().item())
            step_diff_losses.append(loss_diff.detach().item())
            step_out_kd_losses.append(loss_out_kd.detach().item())
            step_wav_kd_losses.append(loss_wav_kd.detach().item())
            epoch_total_losses.append(loss.detach().item())

            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, "
                f"Global Step: {global_step:07d}, "
                f"Loss: {loss.item():.6f}, "
                f"Diff: {loss_diff.item():.6f}, "
                f"OutKD: {loss_out_kd.item():.6f}, "
                f"WavKD: {loss_wav_kd.item():.6f}"
            )

            # Log scalar losses
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                avg_total = (
                    accelerator.gather(
                        torch.tensor(step_total_losses, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_diff = (
                    accelerator.gather(
                        torch.tensor(step_diff_losses, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_out_kd = (
                    accelerator.gather(
                        torch.tensor(step_out_kd_losses, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_wav_kd = (
                    accelerator.gather(
                        torch.tensor(step_wav_kd_losses, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                step_total_losses.clear()
                step_diff_losses.clear()
                step_out_kd_losses.clear()
                step_wav_kd_losses.clear()

                if accelerator.is_main_process:
                    writer.add_scalar("loss/total_step", avg_total, global_step)
                    writer.add_scalar("loss/diff_step", avg_diff, global_step)
                    writer.add_scalar("loss/output_kd_step", avg_out_kd, global_step)
                    writer.add_scalar("loss/wavelet_kd_step", avg_wav_kd, global_step)

                    if use_wandb:
                        wandb.log(
                            {
                                "global_step": global_step,
                                "loss/total_step": avg_total,
                                "loss/diff_step": avg_diff,
                                "loss/output_kd_step": avg_out_kd,
                                "loss/wavelet_kd_step": avg_wav_kd,
                                "train/lr": opt.param_groups[0]["lr"],
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    print(f"save checkpoint to {ckpt_path}")

                    if use_wandb:
                        wandb.log(
                            {
                                "global_step": global_step,
                                "train/checkpoint_step": global_step,
                            },
                            step=global_step,
                        )

            # Optional image logging.
            # 預設關閉，避免訓練時又跑 50-step sampling 爆 VRAM。
            if (
                enable_image_log
                and (global_step % cfg.train.image_every == 0 or global_step == 1)
            ):
                N = min(4, gt.shape[0])

                log_clean = clean[:N]
                log_teacher = teacher_img[:N]
                log_gt = gt[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_prompt = prompt[:N]

                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )

                    if accelerator.is_main_process:
                        images_to_log = [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/condition", log_clean),
                            ("image/teacher", (log_teacher + 1) / 2),
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/condition_aug_decoded",
                                (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                        ]

                        for tag, image in images_to_log:
                            writer.add_image(
                                tag,
                                make_grid(image, nrow=2),
                                global_step,
                            )
                cldm.train()

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1

        avg_epoch_loss = (
            accelerator.gather(
                torch.tensor(epoch_total_losses, device=device).unsqueeze(0)
            )
            .mean()
            .item()
        )
        epoch_total_losses.clear()

        if accelerator.is_main_process:
            writer.add_scalar("loss/total_epoch", avg_epoch_loss, global_step)

            if use_wandb:
                wandb.log(
                    {
                        "global_step": global_step,
                        "loss/total_epoch": avg_epoch_loss,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

    if accelerator.is_main_process:
        print("done!")
        writer.close()

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
