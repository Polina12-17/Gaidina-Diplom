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
        if target.size(1) != 1:
            raise ValueError("Expected target to have 1 channel, but got {} channels".format(target.size(1)))

        with torch.no_grad():
            debayer_pred = f(pred)
            debayer_target = f(target)

        # Проверка на правильное количество каналов после дебайеринга
        if debayer_pred.size(1) != 3:
            raise ValueError("Expected debayer_pred to have 3 channels, but got {} channels".format(debayer_pred.size(1)))
        if debayer_target.size(1) != 3:
            raise ValueError("Expected debayer_target to have 3 channels, but got {} channels".format(debayer_target.size(1)))

        return self.l1_loss(debayer_pred, debayer_target)