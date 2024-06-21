import os

import piq
import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from model import UnetTMO
from source.currentDebayer import debayer
from source.myLoss import TVLoss


def save_image(im, p):
    base_dir = os.path.split(p)[0]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torchvision.utils.save_image(im, p.replace(".CR2", ".jpg"))


@MODEL_REGISTRY
class PSENet(LightningModule):

    def __init__(self, tv_w, gamma_lower, gamma_upper, number_refs, lr, afifi_evaluation=False):
        super().__init__()
        self.tv_w = tv_w
        self.gamma_lower = gamma_lower
        self.gamma_upper = gamma_upper
        self.number_refs = number_refs
        self.afifi_evaluation = afifi_evaluation
        self.lr = lr
        self.model = UnetTMO()
        self.mse = torch.nn.MSELoss()
        self.tv = TVLoss()
        self.l1 = torch.nn.SmoothL1Loss()
        self.saved_input = None
        self.saved_gt = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     betas=[0.9, 0.99])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                               factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss"}

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(
                self.trainer.callback_metrics["total_loss"])

    def training_step(self, batch, batch_idx):
        nth_input, nth_gt = batch

        self.saved_input = nth_input
        self.saved_gt = nth_gt
        im = self.saved_input[0]
        pred_im, pred_gamma = self.model(im)
        img_gt = self.saved_gt
        debayered_img = debayer(pred_im)

        tv_loss = self.tv(debayer(pred_gamma))
        reconstruction_loss = self.l1(debayered_img, img_gt)

        loss = reconstruction_loss + tv_loss * self.tv_w

        # logging
        self.log("train_loss/reconstruction", reconstruction_loss, on_epoch=True, on_step=False)
        self.log("total_loss", loss, on_epoch=True, on_step=False)


        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            pred_im, pred_gamma = self.model(
                batch)
            self.logger.experiment.add_images("val_input", batch, self.current_epoch)
            self.logger.experiment.add_images("val_output", pred_im, self.current_epoch)

    def test_step(self, batch, batch_idx, test_idx=0):
        input_im, path = batch[0], batch[-1]
        pred_im, pred_gamma = self.model(input_im)
        for i in range(len(path)):
            save_image(pred_im[i], os.path.join(self.logger.log_dir, path[i]))

        if len(batch) == 3:
            gt = batch[1]
            psnr = piq.psnr(pred_im, gt)
            ssim = piq.ssim(pred_im, gt)
            self.log("psnr", psnr, on_step=False, on_epoch=True)
            self.log("ssim", ssim, on_step=False, on_epoch=True)
            if self.afifi_evaluation:
                assert len(path) == 1, "only support with batch size 1"
                if "N1." in path[0]:
                    self.log("psnr_under", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_under", ssim, on_step=False, on_epoch=True)
                else:
                    self.log("psnr_over", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_over", ssim, on_step=False, on_epoch=True)
