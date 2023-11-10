from PIL import Image, ImageOps
import numpy as np

def preprocess_image(image, target_size=(28, 28)):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
