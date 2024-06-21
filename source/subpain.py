import torch
from model import UnetTMO

model = UnetTMO()
state_dict = torch.load(".\\pretrained\\last.ckpt", map_location=torch.device('cpu'))["state_dict"]

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_state_dict[k[len("model."):]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)