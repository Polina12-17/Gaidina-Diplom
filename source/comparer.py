import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import piq

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized between 0 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    return piq.ssim(img1, img2, data_range=1.0).item()

def calculate_rmse(img1, img2):
    mse = F.mse_loss(img1, img2)
    rmse = torch.sqrt(mse)
    return rmse.item()

def load_image(filepath):
    image = Image.open(filepath).convert('RGB')
    transform = transforms.ToTensor()
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor_image

# Загрузка изображений
img1 = load_image('a.png')

print(f"img1 min: {torch.min(img1)}")
print(f"img1 max: {torch.max(img1)}")


img2 = load_image('b.png')

# Вычисление метрик
psnr_value = calculate_psnr(img1, img2)
ssim_value = calculate_ssim(img1, img2)
rmse_value = calculate_rmse(img1, img2)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
print(f"RMSE: {rmse_value}")