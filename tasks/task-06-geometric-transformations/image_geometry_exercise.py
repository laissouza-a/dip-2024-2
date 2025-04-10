# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def translate(img: np.ndarray, tx: int = 10, ty: int = 10) -> np.ndarray:
    h, w = img.shape
    translated = np.zeros_like(img)
    translated[ty:h, tx:w] = img[0:h-ty, 0:w-tx]
    return translated

def rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
    return np.flipud(img.T)

def stretch_horizontal(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    h, w = img.shape
    new_w = int(w * scale)
    stretched = np.zeros((h, new_w), dtype=img.dtype)
    for y in range(h):
        for x in range(new_w):
            orig_x = int(x / scale)
            if orig_x < w:
                stretched[y, x] = img[y, orig_x]
    return stretched

def mirror_horizontal(img: np.ndarray) -> np.ndarray:
    return np.fliplr(img)

def barrel_distort(img: np.ndarray, k: float = 0.0005) -> np.ndarray:
    h, w = img.shape
    cx, cy = w / 2, h / 2
    distorted = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            dx = (x - cx) / cx
            dy = (y - cy) / cy
            r = np.sqrt(dx**2 + dy**2)
            factor = 1 + k * r**2

            src_x = int(cx + dx * factor * cx)
            src_y = int(cy + dy * factor * cy)

            if 0 <= src_x < w and 0 <= src_y < h:
                distorted[y, x] = img[src_y, src_x]

    return distorted

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        "translated": translate(img),
        "rotated": rotate_90_clockwise(img),
        "stretched": stretch_horizontal(img),
        "mirrored": mirror_horizontal(img),
        "distorted": barrel_distort(img)
    }
img_path = 'panda.jpg' 
img = Image.open(img_path).convert('L') 

img_np = np.array(img).astype(np.float32) / 255.0  

results = apply_geometric_transformations(img_np)

titles = list(results.keys())
images = list(results.values())

plt.figure(figsize=(12, 6))
for i, (title, image) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()