import cv2
import torch
from debayer import Debayer5x5

f = Debayer5x5().cuda()


class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()  # Определим L1Loss

    def forward(self, pred, target):
        if pred.size(1) != 1:
            raise ValueError("Expected pred to have 1 channel, but got {} channels".format(pred.size(1)))

        with torch.no_grad():
            debayer_pred = f(pred)

        return self.l1_loss(debayer_pred, target)
