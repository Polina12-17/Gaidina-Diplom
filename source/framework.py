import os

import piq  # для оценки изображения
import torch
import torchvision
from iqa import IQA
from loss import TVLoss
from model import UnetTMO
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


def save_image(im, p):
    base_dir = os.path.split(p)[0]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torchvision.utils.save_image(im, p.replace(".CR2", ".jpg"))


@MODEL_REGISTRY
class PSENet(LightningModule):  # нейронка для ??? сегментации текста на изображениях
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
        self.iqa = IQA()  # алгоритм оценки качества изображений todo
        self.saved_input = None
        self.saved_pseudo_gt = None

    def configure_optimizers(self):  # переопределения функции родителя
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     betas=[0.9, 0.99])  # это то что обучает нейронку и меняет в ней веса
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5,
                                                               verbose=True)  # динамически изменять скорость обучения в зависимости от поведения функции потерь. Если в течение определенного количества эпох (указано параметром patience) значение функции потерь не уменьшится, скорость обучения будет уменьшена в factor раз.
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss"}

    def training_epoch_end(self, outputs):  # переопределения функции родителя . вызывается в конце каждой
        # эпохи обучения и позволяет выполнить какие-либо дополнительные действия или обработку результатов
        # обучения перед переходом к следующей эпохе. В данном случае, метод используется для обновления
        # скорости обучения в планировщике скорости обучения
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(
                self.trainer.callback_metrics["total_loss"])  ## адаптирует шаг обучения в зависимости от функции потерь

    def generate_pseudo_gt(self, im):
        bs, ch, h, w = im.shape
        assert ch == 1, "This function expects single-channel images."

        underexposed_ranges = torch.linspace(0, self.gamma_upper, steps=self.number_refs + 1).to(im.device)[:-1]
        step_size = self.gamma_upper / self.number_refs
        underexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device) * step_size + underexposed_ranges[None, :]
        )

        overrexposed_ranges = torch.linspace(self.gamma_lower, 0, steps=self.number_refs + 1).to(im.device)[:-1]
        step_size = - self.gamma_lower / self.number_refs
        overrexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device) * step_size + overrexposed_ranges[None, :]
        )

        gammas = torch.cat([underexposed_gamma, overrexposed_gamma], dim=1)

        synthetic_references = 1 - (1 - im[:, None]) ** gammas[:, :, None, None, None]

        previous_iter_output = self.model(im)[0].clone().detach()
        references = torch.cat([im[:, None], previous_iter_output[:, None], synthetic_references], dim=1)

        nref = references.shape[1]
        scores = self.iqa(references.view(bs * nref, ch, h, w))

        scores = scores.view(bs, nref, 1, h, w)
        max_idx = torch.argmax(scores, dim=1)
        max_idx = max_idx.repeat(1, ch, 1, 1)[:, None]
        pseudo_gt = torch.gather(references, 1, max_idx)
        return pseudo_gt.squeeze(1)

    def training_step(self, batch, batch_idx):  # определяет шаг обучения модели во время каждой эпохи обучения

        nth_input = batch  # Получает текущий ввод
        nth_pseudo_gt = self.generate_pseudo_gt(batch)  # генерирует псевдо-целевое изображение с помощью метода
        if self.saved_input is not None:  # не работает для первой итерации
            # Вычисление потерь и обновление весов модели
            im = self.saved_input
            # Вычисляет потери реконструкции и регуляризации, а затем вычисляет общую потерю
            pred_im, pred_gamma = self.model(im)
            pseudo_gt = self.saved_pseudo_gt
            reconstruction_loss = self.mse(pred_im, pseudo_gt)
            tv_loss = self.tv(pred_gamma)
            loss = reconstruction_loss + tv_loss * self.tv_w

            # logging
            self.log(
                "train_loss/", {"reconstruction": reconstruction_loss, "tv": tv_loss}, on_epoch=True, on_step=False
            )
            self.log("total_loss", loss, on_epoch=True, on_step=False)
            if batch_idx == 0:
                # Если текущий пакет является первым в эпохе (batch_idx == 0),
                # визуализирует входное изображение, псевдо-целевое изображение и выход модели с помощью TensorBoard
                visuals = [im, pseudo_gt, pred_im]
                visuals = torchvision.utils.make_grid([v[0] for v in visuals])
                self.logger.experiment.add_image("images", visuals, self.current_epoch)
        else:
            loss = None
            self.log("total_loss", 0, on_epoch=True, on_step=False)
        # Сохраняет текущий ввод и псевдо-целевое изображение для использования на следующей итерации
        self.saved_input = nth_input
        self.saved_pseudo_gt = nth_pseudo_gt
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


'''
Оценка качества предсказаний: Если в пакете данных (batch) 
также присутствуют настоящие метки (gt), то вычисляются метрики качества, 
такие как PSNR и SSIM, сравнивая предсказанные изображения с настоящими метками.
 Затем значения метрик логируются с помощью метода self.log.
 
При необходимости оценка качества на недо- и переэкспонированных 
изображениях: Если включен режим afifi_evaluation, проверяется, 
принадлежит ли изображение к недо- или переэкспонированным, и затем вычисляются 
и логируются метрики качества для каждого типа изображений.
'''
