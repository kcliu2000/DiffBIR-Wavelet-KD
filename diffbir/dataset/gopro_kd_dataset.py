import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def to_hwc_tensor_01(img):
    return torch.from_numpy(img).contiguous().float()


def to_hwc_tensor_m11(img):
    return torch.from_numpy(img * 2.0 - 1.0).contiguous().float()


class GoProKDDataset(Dataset):
    """
    list format:
    condition_path sharp_path teacher_path
    """

    def __init__(
        self,
        list_path,
        crop_size=512,
        random_crop=True,
        random_flip=True,
        prompt="",
    ):
        self.items = []
        with open(list_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Each line must have 3 paths, got {len(parts)}: {line}")
                self.items.append(parts)

        self.crop_size = crop_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.prompt = prompt

    def __len__(self):
        return len(self.items)

    def _crop(self, cond, gt, teacher):
        h, w, _ = gt.shape
        cs = self.crop_size

        if h < cs or w < cs:
            raise ValueError(f"Image too small: {h}x{w}, crop_size={cs}")

        if self.random_crop:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top = (h - cs) // 2
            left = (w - cs) // 2

        cond = cond[top:top + cs, left:left + cs, :]
        gt = gt[top:top + cs, left:left + cs, :]
        teacher = teacher[top:top + cs, left:left + cs, :]
        return cond, gt, teacher

    def _augment(self, cond, gt, teacher):
        if self.random_flip and random.random() < 0.5:
            cond = np.flip(cond, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
            teacher = np.flip(teacher, axis=1).copy()
        return cond, gt, teacher

    def __getitem__(self, idx):
        cond_path, gt_path, teacher_path = self.items[idx]

        cond = read_rgb(cond_path)
        gt = read_rgb(gt_path)
        teacher = read_rgb(teacher_path)

        cond, gt, teacher = self._crop(cond, gt, teacher)
        cond, gt, teacher = self._augment(cond, gt, teacher)

        return {
            "cond": to_hwc_tensor_01(cond),
            "gt": to_hwc_tensor_m11(gt),
            "teacher": to_hwc_tensor_m11(teacher),
            "prompt": self.prompt,
            "cond_path": cond_path,
            "gt_path": gt_path,
            "teacher_path": teacher_path,
        }