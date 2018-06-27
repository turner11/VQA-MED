
# coding: utf-8

# In[1]:


import os
import numpy as np
from enum import Enum
import time
import datetime
import pandas as pd
from collections import namedtuple
from vqa_logger import logger
from utils.os_utils import File
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)


import keras.layers as keras_layers


def get_time_stamp():
    now = time.time()
    ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
    return ts


# In[2]:


# The location to dump models to
vqa_models_folder          = "C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models"


# In[3]:


# TODO: Duplicate:
class ClassifyStrategies(Enum):
    NLP = 1
    CATEGORIAL = 2
    
# classify_strategy = ClassifyStrategies.CATEGORIAL
classify_strategy = ClassifyStrategies.NLP


# In[4]:


# TODO: Duplicate:
spacy_emmbeding_dim = 384

embedding_dim = 384

# input_dim : the vocabulary size. This is how many unique words are represented in your corpus.
# output_dim : the desired dimension of the word vector. For example, if output_dim = 100, then every word will be mapped onto a vector with 100 elements, whereas if output_dim = 300, then every word will be mapped onto a vector with 300 elements.
# input_length : the length of your sequences. For example, if your data consists of sentences, then this variable represents how many words there are in a sentence. As disparate sentences typically contain different number of words, it is usually required to pad your sequences such that all sentences are of equal length. The keras.preprocessing.pad_sequence method can be used for this (https://keras.io/preprocessing/sequence/).
input_length = 32 # longest question / answer was 28 words. Rounding up to a nice round number

# ATTN - nlp vector: Arbitrary  selected for both question and asnwers
embedded_sentence_length = input_length * embedding_dim 


# In[5]:


DEFAULT_IMAGE_WIEGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}


# In[6]:


#Available merge strategies:
# keras_layers.multiply, keras_layers.add, keras_layers.concatenate, 
# keras_layers.average, keras_layers.co, keras_layers.dot, keras_layers.maximum
            
merge_strategy = keras_layers.concatenate


# In[7]:


#TODO: Duplicate
VqaSpecs = namedtuple('VqaSpecs',['embedding_dim', 'seq_length', 'data_location','meta_data'])
vqa_specs_location = os.path.abspath('./data/vqa_specs.json')


# In[8]:


vqa_specs = File.load_pickle(vqa_specs_location)
meta_data = vqa_specs.meta_data


# Before we start, just for making sure, lets clear the session:

# In[9]:


from keras import backend as keras_backend
keras_backend.clear_session()


# ### Creating the model

# #### The functions the gets the model:

# Define how to build the word-to vector branch:

# In[10]:


def word_2_vec_model(input_tensor):
        # notes:
        # num works: scalar represents size of original corpus
        # embedding_dim : dim reduction. every input string will be encoded in a binary fashion using a vector of this length
        # embedding_matrix (AKA embedding_initializers): represents a pre trained network

        LSTM_UNITS = 512
        DENSE_UNITS = 1024
        DENSE_ACTIVATION = 'relu'


        # logger.debug("Creating Embedding model")
        # x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length,trainable=False)(input_tensor)
        # x = LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(seq_length, embedding_dim))(x)
        # x = BatchNormalization()(x)
        # x = LSTM(units=LSTM_UNITS, return_sequences=False)(x)
        # x = BatchNormalization()(x)
        x= input_tensor # Since using spacy
        x = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(x)
        model = x
        logger.debug("Done Creating Embedding model")
        return model


# In the same manner, define how to build the image representation branch:

# In[11]:


from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D#, Input, Dropout
def get_image_model(base_model_weights=DEFAULT_IMAGE_WIEGHTS, out_put_dim=1024):
    base_model_weights = base_model_weights

    # base_model = VGG19(weights=base_model_weights,include_top=False)
    base_model = VGG19(weights=base_model_weights, include_top=False)
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D(name="image_model_average_pool")(x)
    # let's add a fully-connected layer
    x = Dense(out_put_dim, activation='relu',name="image_model_dense")(x)
    # and a logistic layer -- let's say we have 200 classes
    # predictions = Dense(200, activation='softmax')(x)
    model = x
    
    return base_model.input , model


# And finally, building the model itself:

# In[12]:


model_output_num_units = None
if classify_strategy == ClassifyStrategies.CATEGORIAL:    
    model_output_num_units = len(list(meta_data['ix_to_ans'].keys()) )
elif classify_strategy == ClassifyStrategies.NLP:
    model_output_num_units = embedded_sentence_length    
else:
    raise Exception(f'Unfamilier strategy: {strat}')

logger.debug(f'Model will have {model_output_num_units} output units (Strategy: {classify_strategy})')


# In[13]:


from keras import Model, models, Input, callbacks
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, Embedding, LSTM, BatchNormalization#, GlobalAveragePooling2D, Merge, Flatten

def get_vqa_model(meta):
#     import tensorflow as tf
#     g = tf.Graph()
#     with g.as_default():
    DENSE_UNITS = 1000
    DENSE_ACTIVATION = 'relu'

    OPTIMIZER = 'rmsprop'
    LOSS = 'categorical_crossentropy'
    METRICS = 'accuracy'
    num_classes = len(meta['ix_to_ans'].keys())
    image_model, lstm_model, fc_model = None, None, None
    try:     
        # ATTN:
        lstm_input_tensor = Input(shape=(embedded_sentence_length,), name='embedding_input')
        #lstm_input_tensor = Input(shape=(embedding_dim,), name='embedding_input')

        logger.debug("Getting embedding (lstm model)")
        lstm_model = word_2_vec_model(input_tensor=lstm_input_tensor)

        logger.debug("Getting image model")
        out_put_dim = lstm_model.shape[-1].value
        image_input_tensor, image_model = get_image_model(out_put_dim=out_put_dim)


        logger.debug("merging final model")
        fc_tensors = merge_strategy(inputs=[image_model, lstm_model])
        fc_tensors = BatchNormalization()(fc_tensors)
        fc_tensors = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(fc_tensors)
        fc_tensors = BatchNormalization()(fc_tensors)

        #ATTN:
        fc_tensors = Dense(units=embedded_sentence_length, activation='softmax', name='model_output_sofmax_dense')(fc_tensors)
        #fc_tensors = Dense(units=num_classes, activation='softmax', name='model_output_sofmax_dense')(fc_tensors)

        fc_model = Model(inputs=[lstm_input_tensor, image_input_tensor], output=fc_tensors)
        fc_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])
    except Exception as ex:
        logger.error("Got an error while building vqa model:\n{0}".format(ex))
        models = [(image_model, 'image_model'), (lstm_model, 'lstm_model'), (fc_model, 'lstm_model')]
        for m, name in models:
            if m is not None:
                logger.error("######################### {0} model details: ######################### ".format(name))
                try:
                    m.summary(print_fn=logger.error)
                except Exception as ex2:
                    logger.warning("Failed to print summary for {0}:\n{1}".format(name, ex2))
        raise

    return fc_model

model = get_vqa_model(meta_data)
model


# We better save it:

# In[24]:


import graphviz
import pydot
from keras.utils import plot_model


def print_model_summary_to_file(fn, model):
    # Open the file
    with open(fn,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        

ts = get_time_stamp()

now_folder = os.path.abspath('{0}\\{1}\\'.format(vqa_models_folder, ts))
strat_str = str(classify_strategy).split('.')[-1]
model_name =f'vqa_model_{strat_str}.h5'
model_fn = os.path.join(now_folder,model_name)
model_image_fn = os.path.join(now_folder, 'model_vqa.png')
summary_fn = os.path.join(now_folder, 'model_summary.txt')
logger.debug("saving model to: '{0}'".format(model_fn))

fn_image = os.path.join(now_folder,'model.png')
logger.debug(f"saving model image to {fn_image}")


try:
    File.validate_dir_exists(now_folder)
    model.save(model_fn)  # creates a HDF5 file 'my_model.h5'
    logger.debug("model saved")
    location_message = f"model_location = '{model_fn}'"
except Exception as ex:
    location_message ="Failed to save model:\n{0}".format(ex)
    logger.error(location_message)

try:
    logger.debug("Writing Symmary")
    print_model_summary_to_file(summary_fn, model)
    logger.debug("Done Writing Summary")
    
    logger.debug("Saving image")
    plot_model(model, to_file=fn_image)
    logger.debug(f"Image saved ('{fn_image}')")
#     logger.debug("Plotting model")
#     plot_model(model, to_file=model_image_fn)
#     logger.debug("Done Plotting")
except Exception as ex:
    logger.warning("{0}".format(ex))


# Display a plot + summary:

# In[25]:


# %matplotlib inline
# from matplotlib import pyplot as plt
# %pylab inline


# plt.imshow(img,cmap='gray')
# plt.show()

from IPython.display import Image, display

listOfImageNames = [fn_image]

for imageName in listOfImageNames:
    display(Image(filename=imageName))
model.summary()


# In[27]:


location_message

