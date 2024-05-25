import torch
from model import UnetTMO

# Предполагаем, что модель уже обучена и у вас есть её состояние в переменной `state_dict`
model = UnetTMO()
state_dict = torch.load(".\\pretrained\\last.ckpt", map_location=torch.device('cpu'))["state_dict"]

# Убираем префикс "model." из имен параметров
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_state_dict[k[len("model."):]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)