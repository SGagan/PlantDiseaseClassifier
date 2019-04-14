import io
import base64
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

default_image_size = tuple((256, 256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
def convert_image(image_data):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        if image is not None :  
            image_array = convert_image_to_array(image)
            image_array = np.array(image_array, dtype=np.float16) / 225.0
            return np.expand_dims(image_array, axis=0), None
        else :
            return None, "Error loading image file"
    except Exception as e:
        return None, str(e)