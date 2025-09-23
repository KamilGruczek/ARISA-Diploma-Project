from PIL import Image
import numpy as np

def preprocess_image(image_file, resize=None):
    """Preprocess the image for model prediction."""
    image = Image.open(image_file.stream).convert('RGB')
    if resize:
        image = image.resize(resize)
    return np.array(image)