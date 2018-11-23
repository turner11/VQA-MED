
# coding: utf-8

# ### Training the model

# In[1]:


## VGG all words are Classes (Trainable params: 1,070,916). 'categorical_crossentropy', 'sigmoid' .With f1_score, recall_score, precision_score + accuracy metrics
model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180830_1046_30\\vqa_model_CATEGORIAL.h5'
strategy_str = 'CATEGORIAL'
## VGG all words are Classes (Trainable params: 1,070,916). 'categorical_crossentropy', 'softmax' .With f1_score, recall_score, precision_score + accuracy metrics
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180829_0830_48\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG all words are Classes (Trainable params: 1,070,916) With f1_score, recall_score, precision_score + accuracy metrics
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180828_2149_37\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG all words are Classes (Trainable params: 1,070,916)
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180827_1502_41\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG 2 Classes (Trainable params: 165,762)
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180814_2035_20\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG 4 Classes
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180730_0648_46\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180728_2248_02\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

# ## Resnet 50: 
# trained_model_location = 'C:\Users\Public\Documents\Data\2018\vqa_models\20180730_0524_48\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5'
# loss: 0.1248 - acc: 0.9570 - val_loss: 2.7968 - val_acc: 0.5420
# Training Model: 12:22:54.619203                


# ### Preparing the data for training

# In[2]:


# %%capture
import os
import numpy as np
import pandas as pd
from pandas import HDFStore
from vqa_logger import logger 
from enum import Enum

from functools import partial

from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as keras_backend, callbacks


# In[3]:


# %%capture
import IPython
from common.functions import get_highlited_function_code, get_features, concat_row, sentences_to_hot_vector, hot_vector_to_words
from common.constatns import data_location, vqa_models_folder, vqa_specs_location #train_data, validation_data, 
from common.utils import VerboseTimer
from common.settings import classify_strategy
from common.classes import ClassifyStrategies, EarlyStoppingByAccuracy
from common.model_utils import save_model
from common.os_utils import File
from evaluate.statistical import f1_score, recall_score, precision_score
from evaluate.WbssEvaluator import wbss_score


# #### Loading the Model:

# In[6]:


with VerboseTimer("Loading Model"):
    model = load_model(model_location, custom_objects= {'f1_score': f1_score, 'recall_score':recall_score, 'precision_score':precision_score})
    


# #### Loading the data:

# In[ ]:


logger.debug(f"Loading the data from {data_location}")
with VerboseTimer("Loading Data"):
    with HDFStore(data_location) as store:
        df_data = store['data']  


# In[ ]:


vqa_specs = File.load_pickle(vqa_specs_location)
meta_data_location = vqa_specs.meta_data_location


df_meta_answers = pd.read_hdf(meta_data_location,'answers')
df_meta_words = pd.read_hdf(meta_data_location,'words')
df_meta_imaging_devices = pd.read_hdf(meta_data_location,'imaging_devices')
df_meta_answers.tail(2)


# In[ ]:


logger.debug(f"df_data Shape: {df_data.shape}")
df_data.head(2)


# #### Packaging the data to be in expected input shape

# ##### It makes no sense to train on imageing devices we don't know thier lables

# In[ ]:


# ATTN: 
cols_to_remove = ['both', 'unknown']
def filter_out_unknown_devices(df):
    valid_devices = df_meta_imaging_devices.imaging_device.values
    return df[df.imaging_device.isin(valid_devices)]


df_data_orig = df_data 
df_data = filter_out_unknown_devices(df_data)


# In[ ]:


data_train = df_data[df_data.group == 'train'].copy().reset_index()
data_val = df_data[df_data.group == 'validation'].copy().reset_index()

# print(f'groups:\n{df_data.group.drop_duplicates()}')
# print(len(df_data))
# data_val.head()


# ##### The functions for getting the features & labels:

# In[ ]:


from common.functions import get_features, concat_row
code_get_features = get_highlited_function_code( get_features, remove_comments=True)
code_concat = get_highlited_function_code(concat_row, remove_comments=True)
IPython.display.display(code_get_features)
IPython.display.display(code_concat)


# #### Defining how to get NLP labels

# In[ ]:


def get_nlp_labels():
    labels =  concat_row(df, 'answer_embedding')
    return labels


# #### Defining how to get Categorial fetaures / labels

# In[ ]:


df = data_train

class_df = df_meta_words
class_count = len(class_df)
# class_df

classes_indices_df = [class_df.loc[class_df.word.isin(ans.lower().split())] for ans in  df.answer]
classes_indices = [list(d.index) for d in classes_indices_df]

idx_sample = 9
print(df.answer[idx_sample])
classes_indices[idx_sample]


# ### Will transform the sentences into vector and back using the following:

# In[ ]:


code = get_highlited_function_code(sentences_to_hot_vector,remove_comments=False)
IPython.display.display(code)    

code = get_highlited_function_code(hot_vector_to_words,remove_comments=False)
IPython.display.display(code)  


# #### Check it looks sane by inversing the binarizing:

# In[ ]:


words = df_meta_words.word
sentences =  data_train.answer

arr_one_hot_vector = sentences_to_hot_vector(sentences, words)
categorial_labels = arr_one_hot_vector

idx = 100
answer =  data_train.answer.loc[idx]
print(f'The sentence:\n{answer}')

one_hot_vector = arr_one_hot_vector[idx]
label_words = hot_vector_to_words(one_hot_vector, words)
print('\n\nThe highlighed labels:')
label_words


# In[ ]:


if classify_strategy == ClassifyStrategies.CATEGORIAL:        
    get_labels = partial(sentences_to_hot_vector, words_df=df_meta_words.word)            
    
# elif classify_strategy == ClassifyStrategies.NLP:   
#     get_labels = get_nlp_features_and_labels    

# Note: The shape of answer (for a single recored ) is (number of words, 384)
else:
    raise Exception(f'Unfamilier strategy: {strat}')
print(f'classify stratagy: {classify_strategy}')

with VerboseTimer('Getting train features'):
    features_t = get_features(data_train)   
with VerboseTimer('Getting train labels'):
    labels_t = get_labels(data_train.answer)        
    
with VerboseTimer('Getting train features'):
    features_val = get_features(data_val)
with VerboseTimer('Getting validation labels'):
    labels_val = get_labels(data_val.answer)

# len(features_t[1])


# In[ ]:


validation_input = (features_val, labels_val)


# In[ ]:


print(f'Expectedt shape: {model.input_shape}')
print('---------------------------------------------------------------------------')
print(f'Actual training shape:{features_t[0].shape, features_t[1].shape}')
print(f'Train Labels shape:{labels_t.shape}')
print('---------------------------------------------------------------------------')
print(f'Actual Validation shape:{features_val[0].shape, features_val[1].shape}')
print(f'Validation Labels shape:{labels_val.shape}')


# In[ ]:


# from utils.gpu_utils import test_gpu
# test_gpu()


# In[ ]:


from keras.utils import plot_model
# EPOCHS=25
# BATCH_SIZE = 20

EPOCHS= 1
BATCH_SIZE = 75

# train_features = image_name_question
# validation_input = (validation_features, categorial_validation_labels)

## construct the image generator for data augmentation
# aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#                                horizontal_flip=True, fill_mode="nearest")
# train_generator = aug.flow(train_features, categorial_train_labels)

# stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,mode='auto')
history = None
try:
#     history = model.fit_generator(train_generator,
#                                   validation_data=validation_input,
#                                   steps_per_epoch=len(train_features) // self.batch_size,
#                                   epochs=self.epochs,
#                                   verbose=1,
#                                   callbacks=[stop_callback],
#                                   class_weight=class_weight
#                                   )
    # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


    import tensorflow as tf
    import keras.backend.tensorflow_backend as ktf


    def get_session(gpu_fraction=0.333):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=0, verbose=1,mode='auto')
    acc_early_stop = EarlyStoppingByAccuracy(monitor='accuracy', value=0.98, verbose=1)
    
    tensor_log_dir = os.path.abspath(os.path.join('.','tensor_board_logd'))
    File.validate_dir_exists(tensor_log_dir )
    tensor_board_callback = callbacks.TensorBoard(log_dir=tensor_log_dir )
    callbacks = [stop_callback, acc_early_stop, tensor_board_callback ]  

    with VerboseTimer("Training Model"):
#         with get_session() as sess:
#             ktf.set_session(sess)
#             sess.run(tf.global_variables_initializer())
            
        history = model.fit(features_t,labels_t,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=validation_input,
                            shuffle=True,
                            callbacks=callbacks)
#             sess.close()
            
except Exception as ex:
    logger.error("Got an error training model: {0}".format(ex))
    raise


# ### Save trained model:

# In[ ]:


with VerboseTimer("Saving trained Model"):
    name_suffix = f'{classify_strategy}_trained'
    model_fn, summary_fn, fn_image, fn_history = save_model(model, vqa_models_folder, name_suffix=name_suffix, history=history)

msg = f"Summary: {summary_fn}\n"
msg += f"Image: {fn_image}\n"
msg += f'History: {fn_history or "NONE"}\n' 
location_message = f"model_location = '{model_fn}'"



print(msg)
print(location_message)


# In[ ]:


print (location_message.replace('\\','\\\\'))

