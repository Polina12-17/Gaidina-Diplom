

import data
import framework
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY, LightningCLI


class CustomLightningCLI(LightningCLI): ## класс для работы с аргументами командной строки
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--pipeline", choices=["full", "train", "test"])
        parser.add_argument("--exp_name")
        parser.add_argument("--version")
        parser.add_argument("--checkpoint")


def main():
    MODEL_REGISTRY.register_classes(framework, pl.core.lightning.LightningModule)  # регистрируем нашу модель
    DATAMODULE_REGISTRY.register_classes(data, pl.core.LightningDataModule)  # регистрируем наши данные

    cli = CustomLightningCLI(
        auto_registry=True,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_overwrite=True,
        run=False,
        trainer_defaults={
            "callbacks": ModelCheckpoint(
                filename="{epoch:02d}-{val_metric:.2f}",
                every_n_epochs=3,
                save_last=True,
            )
        },
    )
    '''
    auto_registry=True: Включает автоматическую регистрацию моделей и других компонентов.
    subclass_mode_model=True: Включает режим подкласса для определения моделей. Это означает, что классы моделей будут определяться как подклассы LightningModule.
    subclass_mode_data=True: Аналогично, включает режим подкласса для определения данных.
    save_config_overwrite=True: Перезаписывает конфигурацию сохранения моделей, если она уже существует.
    run=False: Не запускает CLI сразу после создания экземпляра. Это означает, что пользователю нужно будет явно вызвать метод запуска CLI.
    trainer_defaults: Параметры по умолчанию для тренера (обучателя). В данном случае, это включает коллбэк ModelCheckpoint, который будет сохранять модели на диске каждые 10 эпох и сохранять последнюю эпоху.
    '''
    print(cli.trainer.default_root_dir)

    cli.trainer.logger = pl.loggers.TensorBoardLogger(  # настройка логгера
        save_dir=cli.trainer.default_root_dir,
        name=cli.config["exp_name"],
        version=cli.config["version"],
        default_hp_metric=False,
    )

    if cli.config["pipeline"] == "full":
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config["checkpoint"])  #fit - обучение todo проверить как оно работает
        cli.trainer.test(
            cli.model,
            cli.datamodule,
            # ckpt_path='best'
        )
    elif cli.config["pipeline"] == "train":
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config["checkpoint"])
    elif cli.config["pipeline"] == "test":
        cli.trainer.test(cli.model, cli.datamodule, cli.config["checkpoint"])
if __name__ == "__main__":
        main()