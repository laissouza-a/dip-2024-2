import numpy as np

def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """

    height, width = image.shape[:2]  # Get dimensions
    dtype = image.dtype  # Data type
    depth = image.shape[2] if image.ndim == 3 else 1  # Number of channels (depth)
    
    min_val = np.min(image)  # Minimum pixel value
    max_val = np.max(image)  # Maximum pixel value
    mean_val = np.mean(image)  # Mean pixel value
    std_val = np.std(image)  # Standard deviation of pixel values

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }

# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")
