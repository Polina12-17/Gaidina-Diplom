import cv2
import torch

from source.debayer import debayer


class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        self.l1_loss = torch.nn.SmoothL1Loss()  # Определим L1Loss

    def forward(self, pred, raw, target):
        deb =debayer(pred, raw)

        assert isinstance(deb, torch.Tensor), f"Expected input to be a tensor, but got {type(deb)}"
        assert isinstance(target, torch.Tensor), f"Expected target to be a tensor, but got {type(target)}"
        assert deb.size() == target.size(), f"Expected input and target to have the same size, but got {deb.size()} and {target.size()}"

        return self.l1_loss(deb, target)
