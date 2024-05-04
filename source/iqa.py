import torch
import torch.nn as nn

'''
Этот модуль представляет собой метод для оценки качества изображений,
который может использоваться в задачах компьютерного зрения для
измерения важных аспектов изображений, таких как насыщенность,
контраст и экспозиция.
'''
class IQA(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        ps = 25 # Размер окна для выполнения среднего размытия
        self.exposed_level = 0.5 #  Уровень экспозиции, который используется для вычисления показателя освещенности.
        self.mean_pool = torch.nn.Sequential(torch.nn.ReflectionPad2d(ps // 2), torch.nn.AvgPool2d(ps, stride=1)) # Последовательность слоев PyTorch, которая
        # выполняет операцию среднего размытия над изображениями

    def forward(self, images):
        eps = 1 / 255.0 #  Небольшое число, используемое для предотвращения деления на ноль
        max_rgb = torch.max(images, dim=1, keepdim=True)[0] # Максимальное и минимальное значение RGB для каждого пикселя в пакете изображений
        min_rgb = torch.min(images, dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb + eps) / (max_rgb + eps) # Вычисление насыщенности каждого пикселя в изображениях на основе формулы насыщенности

        mean_rgb = self.mean_pool(images).mean(dim=1, keepdim=True) # Среднее значение RGB для каждого пикселя
        # в пакете изображений, вычисленное с использованием среднего размытия
        exposedness = torch.abs(mean_rgb - self.exposed_level) + eps #  Показатель экспозиции для каждого пикселя в
        # изображениях, вычисленный как абсолютное отклонение от уровня экспозиции

        contrast = self.mean_pool(images * images).mean(dim=1, keepdim=True) - mean_rgb**2 # Контраст каждого пикселя в изображениях,
        # вычисленный как разница между средним значением
        # квадратов пикселей и квадратом среднего значения пикселей.
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
