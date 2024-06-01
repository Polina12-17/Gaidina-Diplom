import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Функция для вычисления PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX ** 2) / mse)

# Загрузка изображений
original = cv2.imread('DSC_0495.png', cv2.IMREAD_GRAYSCALE)
compressed = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

# Вычисление PSNR
psnr_value = calculate_psnr(original, compressed)
print(f'PSNR: {psnr_value} dB')

# Вычисление SSIM
ssim_value, _ = ssim(original, compressed, full=True)
print(f'SSIM: {ssim_value}')

# Визуализация (если требуется)
# В данном примере мы просто выведем значения, но можно построить графики.
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(original, cmap='green')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(compressed, cmap='gray')
plt.title('Compressed Image')
plt.show()
