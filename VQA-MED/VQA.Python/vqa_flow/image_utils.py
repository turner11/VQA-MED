import logging
logger = logging.getLogger(__name__)


from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image



def image_to_pre_process_input(image_path, image_size):
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # x = np.array(data, dtype="float") / 255.0
    return x




