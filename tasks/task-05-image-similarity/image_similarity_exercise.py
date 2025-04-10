# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def mse(i1: np.ndarray, i2: np.ndarray) -> float:
    return np.mean((i1 - i2) ** 2)

def psnr(i1: np.ndarray, i2: np.ndarray) -> float:
    mse_val = mse(i1, i2)
    if mse_val == 0:
        return float('inf')
    max_pixel = 1.0 
    return 20 * np.log10(max_pixel / np.sqrt(mse_val))

def ssim(i1: np.ndarray, i2: np.ndarray) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    sigma1 = np.var(i1)
    sigma2 = np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)

    return numerator / denominator

def npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    i1_mean = i1 - np.mean(i1)
    i2_mean = i2 - np.mean(i2)
    numerator = np.sum(i1_mean * i2_mean)
    denominator = np.sqrt(np.sum(i1_mean**2) * np.sum(i2_mean**2))
    return numerator / denominator if denominator != 0 else 0

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }

from PIL import Image
import numpy as np

img_path_1 = 'bear.jpg'
img_path_2 = 'brownBear.jpg'

img1 = Image.open(img_path_1).convert('L')  #grayscale
img2 = Image.open(img_path_2).convert('L')  

img2 = img2.resize(img1.size)

i1 = np.array(img1).astype(np.float32) / 255.0
i2 = np.array(img2).astype(np.float32) / 255.0

results = compare_images(i1, i2)

for metric, value in results.items():
    print(f"{metric.upper()}: {value:.6f}")
