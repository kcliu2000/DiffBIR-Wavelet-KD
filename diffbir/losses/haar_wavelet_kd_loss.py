import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletKDLoss(nn.Module):
    def __init__(self, high_freq_weight=2.0):
        super(HaarWaveletKDLoss, self).__init__()
        self.hf_weight = high_freq_weight

        h0 = torch.tensor([[1/2, 1/2], [1/2, 1/2]]).view(1, 1, 2, 2)
        h1 = torch.tensor([[1/2, 1/2], [-1/2, -1/2]]).view(1, 1, 2, 2)
        h2 = torch.tensor([[1/2, -1/2], [1/2, -1/2]]).view(1, 1, 2, 2)
        h3 = torch.tensor([[1/2, -1/2], [-1/2, 1/2]]).view(1, 1, 2, 2)

        filters = torch.cat([h0, h1, h2, h3], dim=0).repeat(3, 1, 1, 1)
        self.register_buffer("filters", filters)

    def forward(self, student_img, teacher_img):
        """
        student_img, teacher_img: B, 3, H, W
        兩者 range 要一致，例如都在 [-1, 1] 或都在 [0, 1]
        """

        teacher_img = teacher_img.detach()
        filters = self.filters.to(device=student_img.device, dtype=student_img.dtype)

        stu_wav = F.conv2d(student_img, filters, stride=2, groups=3)
        tea_wav = F.conv2d(teacher_img, filters, stride=2, groups=3)

        stu_LL, stu_H = stu_wav[:, 0:3, :, :], stu_wav[:, 3:, :, :]
        tea_LL, tea_H = tea_wav[:, 0:3, :, :], tea_wav[:, 3:, :, :]

        loss_LL = F.l1_loss(stu_LL, tea_LL)
        loss_H = F.l1_loss(stu_H, tea_H)

        return loss_LL + self.hf_weight * loss_H
