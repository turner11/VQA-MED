from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.models import Model, Sequential


class ImageModelGenerator(object):
    """"""



    _DENSE_UNITS = 1024
    _DENSE_ACTIVATION = 'relu'
    _DROPOUT_RATE= 0.5
    #  Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation

    DEFAULT_WIEGHTS = 'imagenet'
    image_size_by_base_models = {'imagenet': (224, 224)}

    def __init__(self):
        """"""
        super().__init__()

    @classmethod
    def get_image_model(cls,  base_model_weights=None, out_put_dim=1024):
        base_model_weights = base_model_weights or cls.DEFAULT_WIEGHTS

        # base_model = VGG19(weights=base_model_weights,include_top=False)
        base_model = VGG19(weights=base_model_weights, include_top=False)
        base_model.trainable = False

        x = base_model.output
        # add a global spatial average pooling layer
        x = GlobalAveragePooling2D(name="image_model_average_pool")(x)
        # let's add a fully-connected layer
        x = Dense(out_put_dim, activation='relu',name="image_model_dense")(x)
        # and a logistic layer -- let's say we have 200 classes
        # predictions = Dense(200, activation='softmax')(x)
        model = x

        return base_model.input , model


    def __repr__(self):
        return super().__repr__()
