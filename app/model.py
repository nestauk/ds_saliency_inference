import tensorflow as tf
from PIL import Image
import io

imported = tf.saved_model.load("./app")
imported = imported.signatures["serving_default"]

def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image

def predict(input_image):
    """Reads file and returns prediction

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    tensor = tf.io.decode_image(input_image, channels=3)

    inference_shape = (240, 320)
    original_shape = tensor.shape[:2]

    input_tensor = tf.expand_dims(tensor, axis=0)

    input_tensor = tf.image.resize(input_tensor, inference_shape,
                               preserve_aspect_ratio=True)
    saliency = imported(input_tensor)["output"]

    saliency = tf.image.resize(saliency, original_shape)
    return saliency.numpy()[0]
