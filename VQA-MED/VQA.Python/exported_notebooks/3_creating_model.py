
# coding: utf-8

# In[ ]:


# %%capture
import os
import numpy as np
import pandas as pd
from enum import Enum
import time
import datetime
import keras.layers as keras_layers
from pandas import HDFStore

from vqa_logger import logger


# In[ ]:


from common.os_utils import File
from common.settings import classify_strategy, embedded_sentence_length, get_stratagy_str
from common.classes import ClassifyStrategies, VqaSpecs
from common.model_utils import save_model
from common.constatns import vqa_models_folder, vqa_specs_location
from evaluate.statistical import f1_score, recall_score, precision_score
from evaluate.WbssEvaluator import wbss_score


# In[ ]:


DEFAULT_IMAGE_WIEGHTS = 'imagenet'

#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}


# In[ ]:


# categorial_data_frame = 'answers'
categorial_data_frame = 'imaging_devices'
categorial_data_frame = 'words'


# In[ ]:


#Available merge strategies:
# keras_layers.multiply, keras_layers.add, keras_layers.concatenate, 
# keras_layers.average, keras_layers.co, keras_layers.dot, keras_layers.maximum
            
merge_strategy = keras_layers.concatenate


# In[ ]:


vqa_specs = File.load_pickle(vqa_specs_location)
meta_data_location = vqa_specs.meta_data_location


# In[ ]:


print(meta_data_location)
with pd.HDFStore(meta_data_location, 'r') as hdf:
    keys = list(hdf.keys())
    print(f"meta Keys: {keys}")




df_meta_answers = pd.read_hdf(meta_data_location,'answers')
df_meta_words = pd.read_hdf(meta_data_location,'words')
df_meta_imaging_devices = pd.read_hdf(meta_data_location,'imaging_devices')
df_meta_answers.tail(2)


# Before we start, just for making sure, lets clear the session:

# In[ ]:


from keras import backend as keras_backend
keras_backend.clear_session()


# ### Creating the model

# #### The functions the gets the model:

# Define how to build the word-to vector branch:

# In[ ]:


#  Input 0 is incompatible with layer lstm_1: expected ndim=3, found ndim=2
# Input 0 is incompatible with layer embbeding_LSTM_1: expected ndim=3, found ndim=2
from keras.layers import Flatten, LSTM, BatchNormalization
def word_2_vec_model(input_tensor):
        #print(dir(input_tensor))
#         print('---------------------------------------------')
#         print('Tensor shape: {0}'.format(input_tensor.get_shape()))
#         print('---------------------------------------------')
#         print(input_tensor.shape)
#         print('---------------------------------------------')
#         print('embedded_sentence_length: {0}'.format(embedded_sentence_length))
#         print('---------------------------------------------')
#         return
        
        

        LSTM_UNITS = 64 # 512
        DENSE_UNITS = 256
        DENSE_ACTIVATION = 'relu'
        
        logger.debug("Creating Embedding model")
        x= input_tensor # Since using spacy
 
        x = LSTM(units=LSTM_UNITS, return_sequences=False, name='embbeding_LSTM',  input_shape=(1,embedded_sentence_length))(x)
        x = BatchNormalization(name='embbeding_batch_normalization')(x)
        
#         x = LSTM(units=LSTM_UNITS, return_sequences=True, name='embbeding_LSTM_1',  input_shape=(1,embedded_sentence_length))(x)
#         x = BatchNormalization(name='embbeding_batch_normalization_1')(x)
#         x = LSTM(units=LSTM_UNITS, return_sequences=False, name='embbeding_LSTM_2')(x)
#         x = BatchNormalization(name='embbeding_batch_normalization_2')(x)
        
#         x = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(x)
        model = x
        logger.debug("Done Creating Embedding model")
        return model


# In the same manner, define how to build the image representation branch:

# In[ ]:


from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

from keras.layers import Dense, GlobalAveragePooling2D#, Input, Dropout
def get_image_model(base_model_weights=DEFAULT_IMAGE_WIEGHTS, out_put_dim=1024):
    base_model_weights = base_model_weights

    base_model = VGG19(weights=base_model_weights,include_top=False)
#     base_model = ResNet50(weights=base_model_weights, include_top=False)
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D(name="image_model_average_pool")(x)
    
    # x = Dense(out_put_dim, activation='relu',name="image_model_dense")(x)
    
    model = x
    
    return base_model.input , model


# And finally, building the model itself:

# In[ ]:


model_output_num_units = None
if classify_strategy == ClassifyStrategies.CATEGORIAL:    
    model_output_num_units = len(pd.read_hdf(meta_data_location,categorial_data_frame))
elif classify_strategy == ClassifyStrategies.NLP:
    model_output_num_units = embedded_sentence_length    
else:
    raise Exception(f'Unfamilier strategy: {strat}')

logger.debug(f'Model will have {model_output_num_units} output units (Strategy: {classify_strategy}). Categorial coolumn: "{categorial_data_frame}"')


# In[ ]:


from keras import Model, models, Input, callbacks
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, Embedding, LSTM, BatchNormalization, Activation, Flatten#, GlobalAveragePooling2D, Merge, Flatten

def get_vqa_model():
#     import tensorflow as tf
#     g = tf.Graph()
#     with g.as_default():
    DENSE_UNITS = 256
    DENSE_ACTIVATION = 'relu'

    OPTIMIZER = 'rmsprop'
#     LOSS, ACTIVATION = 'categorical_crossentropy', 'softmax' #good for a model to predict multiple mutually-exclusive classes.
#     LOSS, ACTIVATION = 'binary_crossentropy', 'sigmoid'
    LOSS, ACTIVATION = 'categorical_crossentropy', 'sigmoid'
    
    METRICS = [f1_score, recall_score, precision_score, 'accuracy']#[f1_score, 'accuracy']#['accuracy'] #'f1score', 'recall','precision'
    image_model, lstm_model, fc_model = None, None, None
    try:     
        # ATTN:
        lstm_input_tensor = Input(shape=(embedded_sentence_length,1), name='embedding_input')
        #lstm_input_tensor = Input(shape=(embedding_dim,), name='embedding_input')

        logger.debug("Getting embedding (lstm model)")
        lstm_model = word_2_vec_model(input_tensor=lstm_input_tensor)

        logger.debug("Getting image model")
        out_put_dim = lstm_model.shape[-1].value
        image_input_tensor, image_model = get_image_model(out_put_dim=out_put_dim)


        logger.debug("merging final model")
        fc_tensors = merge_strategy(inputs=[image_model, lstm_model])
#         fc_tensors = BatchNormalization()(fc_tensors)
        fc_tensors = Dense(units=DENSE_UNITS)(fc_tensors)
        fc_tensors = BatchNormalization()(fc_tensors)
        fc_tensors = Activation(DENSE_ACTIVATION)(fc_tensors)

        #ATTN:
        fc_tensors = Dense(units=model_output_num_units, activation=ACTIVATION, name='model_output_sofmax_dense')(fc_tensors)        

        fc_model = Model(inputs=[lstm_input_tensor, image_input_tensor], output=fc_tensors)
        fc_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
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

model = get_vqa_model()
model


# ##### We better save it:

# In[ ]:


strategy_str = get_stratagy_str()


model_fn, summary_fn, fn_image, _ = save_model(model, vqa_models_folder, name_suffix=strategy_str)

msg = f"Summary: {summary_fn}\n"
msg += f"Image: {fn_image}\n"
location_message = f"model_location = '{model_fn}'"


print(msg)
print (location_message)


# ### Display a plot + summary:

# #### Where are the trainable parameters?

# In[ ]:


import pandas as pd
K = keras_backend
names_and_trainable_params = {w.name: np.prod(K.get_value(w).shape) for w in model.trainable_weights}
names_and_trainable_params = {(w.name, np.prod(K.get_value(w).shape)) for w in model.trainable_weights}
a = {'layer': [tpl[0] for tpl in names_and_trainable_params],
     'trainable_params': [tpl[1] for tpl in names_and_trainable_params]
    }
df = pd.DataFrame.from_dict(a)
df_sorted = df.sort_values(['trainable_params'], ascending=[False]).reset_index()
df_sorted['pretty_value'] = df_sorted.apply(lambda x: "{:,}".format(x['trainable_params']), axis=1)
top = df_sorted[df_sorted.trainable_params > 1000]
print(f'Got a total of {"{:,}".format(sum(df_sorted.trainable_params))} trainable parameters')
top


# In[ ]:


from IPython.display import Image, display
listOfImageNames = [fn_image]

for imageName in listOfImageNames:
    display(Image(filename=imageName))
model.summary()


# Copy these items to the next notebook of training the model

# In[ ]:


# logger.debug('Done')
print(location_message.replace('\\', '\\\\'))
print(f'strategy_str = \'{strategy_str}\'')

