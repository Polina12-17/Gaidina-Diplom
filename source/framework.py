import os

import piq  # для оценки изображения
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
        self.afifi_evaluation = afifi_evaluation  # выполняется ли оценка с использованием метода Afifi
        self.lr = lr  # скорость обучения
        self.model = UnetTMO()
        self.mse = torch.nn.MSELoss()  # функция потерь todo это шо???
        self.tv = TVLoss()  # регуляризация общей вариации todo
        self.l1 = torch.nn.SmoothL1Loss()  # регуляризация общей вариации todo
        self.saved_input = None
        self.saved_gt = None

    def configure_optimizers(self):  # переопределения функции родителя
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     betas=[0.9, 0.99])  # это то что обучает нейронку и меняет в ней веса
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                               factor=0.5)  # динамически изменять скорость обучения в зависимости от поведения функции потерь. Если в течение определенного количества эпох (указано параметром patience) значение функции потерь не уменьшится, скорость обучения будет уменьшена в factor раз.
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss"}

    def training_epoch_end(self, outputs):  # переопределения функции родителя . вызывается в конце каждой
        # эпохи обучения и позволяет выполнить какие-либо дополнительные действия или обработку результатов
        # обучения перед переходом к следующей эпохе. В данном случае, метод используется для обновления
        # скорости обучения в планировщике скорости обучения
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(
                self.trainer.callback_metrics["total_loss"])  ## адаптирует шаг обучения в зависимости от функции потерь

    def training_step(self, batch, batch_idx):
        nth_input, nth_gt = batch
        if self.saved_input is not None:  # не работает для первой итерации
            im = self.saved_input[0]
            pred_im, pred_gamma = self.model(im)
            img_gt = self.saved_gt
            debayered_img = debayer(pred_im)

            # tv_loss = self.tv(debayer(pred_gamma))
            reconstruction_loss = self.l1(debayered_img, img_gt)

            loss = reconstruction_loss * self.tv_w  # + tv_loss * self.tv_w

            # После одного шага оптимизации, напечатайте значения градиентов
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm()}")
            #     else:
            #         print(f"{name}: No gradient")

            # for name, param in self.model.named_parameters():
            #     if not param.requires_grad:
            #         print(f"{name} does not require grad")
            #     else:
            #         print(f"{name} ok")

            # logging
            self.log("train_loss/reconstruction", reconstruction_loss, on_epoch=True, on_step=False)
            # self.log("train_loss/tv", tv_loss, on_epoch=True, on_step=False)
            self.log("total_loss", loss, on_epoch=True, on_step=False)

        else:
            loss = None
            self.log("total_loss", 0, on_epoch=True, on_step=False)

        self.saved_input = nth_input
        self.saved_gt = nth_gt

        return loss

    def validation_step(self, batch, batch_idx):  # Метод validation_step используется для
        # выполнения шага валидации модели на отдельном пакете данных.
        # Здесь, в отличие от training_step, нет обновления весов модели
        # или вычисления потерь - это просто оценка модели на валидационных данных.
        if batch_idx == 0:
            pred_im, pred_gamma = self.model(
                batch)  # Модель используется для получения предсказанных значений на текущем пакете данных.
            # Здесь pred_im - предсказанные изображения, а
            # pred_gamma - предсказанные значения параметра гамма
            self.logger.experiment.add_images("val_input", batch, self.current_epoch)
            self.logger.experiment.add_images("val_output", pred_im, self.current_epoch)

    def test_step(self, batch, batch_idx, test_idx=0):  # используется для выполнения шага тестирования
        # модели на отдельном пакете данных. Он похож на
        # validation_step, но предназначен для тестирования
        # модели, а не ее валидации.
        input_im, path = batch[0], batch[-1]  # Извлекает входное изображение
        pred_im, pred_gamma = self.model(input_im)  # предсказывает значение
        for i in range(len(path)):
            save_image(pred_im[i], os.path.join(self.logger.log_dir, path[i]))  # сохраняет изображения

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
