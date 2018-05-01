from keras.applications.vgg19 import VGG19
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential


class ImageModelGenerator(object):
    """"""

    __models_by_initial_weights = {}

    _DENSE_UNITS = 1024
    _DENSE_ACTIVATION = 'relu'
    _DROPOUT_RATE= 0.5
    #  Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    IMAGE_SIZE =  (224, 224)

    def __init__(self):
        """"""
        super().__init__()

    @classmethod
    def get_image_model(cls, base_model_weights=None, out_put_dim=1024):
        base_model_weights = base_model_weights or 'imagenet'

        model = cls.__models_by_initial_weights.get(base_model_weights,None)
        if model is None:

            base_model = VGG19(weights=base_model_weights)
            base_model.trainable = False
            model_name = "image_model_{0}".format(base_model_weights)

            vgg_sub_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output, name=model_name)

            vgg_sub_model.trainable = False
            model = Sequential()
            model.add(vgg_sub_model)
            model.add(GlobalAveragePooling2D())
            model.add(Dense(out_put_dim, activation='relu'))
            # model.summary()

            ## This does not work well for multiple models, cannot be cached...
            ## when working only on VQA, if needed uncomment
            # cls.__models_by_initial_weights[base_model_weights] = vgg_sub_model

        return model

    def __repr__(self):
        return super().__repr__()
