import numpy as np
import cv2
import urllib.request


def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv2.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as resp:
        image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)

    # Decodificar a imagem usando OpenCV
    image = cv2.imdecode(image_array, kwargs.get('flags', cv2.IMREAD_COLOR))
    cv2.imshow('Downloaded Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


load_image_from_url('http://answers.opencv.org/upfiles/logo_2.png')
