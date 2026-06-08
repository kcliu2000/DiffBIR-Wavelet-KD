#!/usr/bin/env python3
"""
Evaluate GoPro restoration results with PSNR / SSIM / optional LPIPS.

This script matches restored images and GT images by basename.
It is designed for DiffBIR / GoPro experiments.

Examples:

1) PSNR / SSIM only:
python tools_eval_gopro_metrics.py \
  --result_dir results/gopro_lite_wavelet_30k \
  --gt_dir datasets/GoPro/test/sharp \
  --csv_path results/metrics_gopro_lite_wavelet_30k.csv

2) PSNR / SSIM / LPIPS:
python tools_eval_gopro_metrics.py \
  --result_dir results/gopro_lite_wavelet_30k \
  --gt_dir datasets/GoPro/test/sharp \
  --use_lpips \
  --lpips_net alex \
  --csv_path results/metrics_gopro_lite_wavelet_30k.csv

3) Quick debug with first 5 matched images:
python tools_eval_gopro_metrics.py \
  --result_dir results/test_gopro_lite_wavelet_30k_5imgs \
  --gt_dir datasets/GoPro/test/sharp \
  --max_images 5
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(root: str) -> Dict[str, Path]:
    """Collect images recursively and map basename -> path."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for p in sorted(root_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            name = p.name
            if name in mapping:
                duplicates.setdefault(name, [mapping[name]]).append(p)
            else:
                mapping[name] = p

    if duplicates:
        msg = ["Duplicate basenames found. Please avoid ambiguous matching:"]
        for k, paths in list(duplicates.items())[:20]:
            msg.append(f"  {k}: " + " | ".join(str(x) for x in paths))
        if len(duplicates) > 20:
            msg.append(f"  ... and {len(duplicates) - 20} more duplicates")
        raise RuntimeError("\n".join(msg))

    return mapping


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def center_crop_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center crop larger image(s) so both have same H,W."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    w = min(wa, wb)

    def crop(x: np.ndarray, h: int, w: int) -> np.ndarray:
        hx, wx = x.shape[:2]
        top = max((hx - h) // 2, 0)
        left = max((wx - w) // 2, 0)
        return x[top:top + h, left:left + w]

    return crop(a, h, w), crop(b, h, w)


def shave_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img

    h, w = img.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(f"Image too small for border={border}: {img.shape}")

    return img[border:-border, border:-border]


def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    if HAS_SKIMAGE:
        return float(skimage_ssim(gt, pred, channel_axis=2, data_range=255))

    # Fallback: simple global SSIM over RGB.
    # This is less standard than skimage's implementation but avoids hard failure.
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    vals = []
    for c in range(3):
        x = pred_f[..., c]
        y = gt_f[..., c]
        mux, muy = x.mean(), y.mean()
        sigx, sigy = x.var(), y.var()
        sigxy = ((x - mux) * (y - muy)).mean()
        vals.append(
            ((2 * mux * muy + c1) * (2 * sigxy + c2))
            / ((mux**2 + muy**2 + c1) * (sigx + sigy + c2))
        )
    return float(np.mean(vals))


def to_lpips_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    # RGB uint8 HWC -> BCHW float [-1, 1]
    x = torch.from_numpy(img.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = x * 2.0 - 1.0
    return x.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="Directory of restored/result images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory of GT sharp images.")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to save per-image CSV.")
    parser.add_argument("--border", type=int, default=0, help="Crop border before metrics. Default: 0.")
    parser.add_argument("--center_crop_if_mismatch", action="store_true", help="Center crop if result/GT sizes mismatch.")
    parser.add_argument("--use_lpips", action="store_true", help="Compute LPIPS if lpips package is installed.")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for LPIPS, e.g. cuda or cpu.")
    parser.add_argument("--max_images", type=int, default=-1, help="For quick debug. -1 means all images.")
    args = parser.parse_args()

    result_map = collect_images(args.result_dir)
    gt_map = collect_images(args.gt_dir)

    common_names = sorted(set(result_map.keys()) & set(gt_map.keys()))
    missing_gt = sorted(set(result_map.keys()) - set(gt_map.keys()))
    missing_result = sorted(set(gt_map.keys()) - set(result_map.keys()))

    print(f"result images: {len(result_map)}")
    print(f"gt images:     {len(gt_map)}")
    print(f"matched:       {len(common_names)}")
    print(f"result w/o gt: {len(missing_gt)}")
    print(f"gt w/o result: {len(missing_result)}")

    if len(common_names) == 0:
        raise RuntimeError("No matched images by basename.")

    if missing_gt:
        print("WARNING: some result images do not have GT. First 10:")
        for n in missing_gt[:10]:
            print("  ", n)

    if missing_result:
        print("WARNING: some GT images do not have result. First 10:")
        for n in missing_result[:10]:
            print("  ", n)

    if args.max_images > 0:
        common_names = common_names[: args.max_images]
        print(f"Evaluate first {len(common_names)} matched images due to --max_images")

    lpips_model = None
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.use_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
            lpips_model.eval()
            print(f"LPIPS enabled: net={args.lpips_net}, device={device}")
        except Exception as e:
            raise ImportError(
                "LPIPS requested but failed to import/init. Install with: pip install lpips\n"
                f"Original error: {e}"
            )

    rows = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for idx, name in enumerate(common_names, 1):
        pred_path = result_map[name]
        gt_path = gt_map[name]

        pred = read_rgb(pred_path)
        gt = read_rgb(gt_path)

        if pred.shape[:2] != gt.shape[:2]:
            if args.center_crop_if_mismatch:
                pred, gt = center_crop_to_match(pred, gt)
            else:
                raise ValueError(
                    f"Size mismatch for {name}: result={pred.shape}, gt={gt.shape}. "
                    "Use --center_crop_if_mismatch if this is expected."
                )

        if args.border > 0:
            pred = shave_border(pred, args.border)
            gt = shave_border(gt, args.border)

        psnr = calc_psnr(pred, gt)
        ssim = calc_ssim(pred, gt)

        row = {
            "filename": name,
            "result_path": str(pred_path),
            "gt_path": str(gt_path),
            "psnr": psnr,
            "ssim": ssim,
        }

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        if lpips_model is not None:
            with torch.no_grad():
                pred_t = to_lpips_tensor(pred, device)
                gt_t = to_lpips_tensor(gt, device)
                lp = float(lpips_model(pred_t, gt_t).item())
            row["lpips"] = lp
            lpips_values.append(lp)

        rows.append(row)

        if idx % 100 == 0 or idx == len(common_names):
            print(
                f"[{idx}/{len(common_names)}] "
                f"current mean PSNR={np.mean(psnr_values):.4f}, "
                f"SSIM={np.mean(ssim_values):.4f}"
            )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    print("\n========== Evaluation Summary ==========")
    print(f"result_dir: {args.result_dir}")
    print(f"gt_dir:     {args.gt_dir}")
    print(f"images:     {len(common_names)}")
    print(f"PSNR:       {mean_psnr:.4f}")
    print(f"SSIM:       {mean_ssim:.4f}")
    if lpips_values:
        print(f"LPIPS:      {float(np.mean(lpips_values)):.4f}")
    print("========================================\n")

    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["filename", "result_path", "gt_path", "psnr", "ssim"]
        if lpips_values:
            fieldnames.append("lpips")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"Saved per-image CSV to: {csv_path}")


if __name__ == "__main__":
    main()
