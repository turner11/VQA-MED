from keras.applications.vgg19 import VGG19
from keras.models import Model


class ImageModelGenerator(object):
    """"""

    __models_by_initial_weights = {}

    def __init__(self):
        """"""
        super().__init__()

    @classmethod
    def get_image_model(cls, base_model_weights='imagenet'):
        model = cls.__models_by_initial_weights.get(base_model_weights,None)
        if model is None:
            base_model = VGG19(weights=base_model_weights)
            base_model.trainable = False
            model_name = "image_model_{0}".format(base_model_weights)
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output, name=model_name)
            model.trainable = False
            cls.__models_by_initial_weights[base_model_weights] = model

        return model

    def __repr__(self):
        return super().__repr__()
