
# coding: utf-8

# ### Training the model

# In[1]:


## VGG 2 Classes (Trainable params: 165,762)
model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180814_2035_20\\vqa_model_CATEGORIAL.h5'
strategy_str = 'CATEGORIAL'

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
from pandas import HDFStore
from vqa_logger import logger 
from enum import Enum

from functools import partial

from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as keras_backend


# In[3]:


get_ipython().run_cell_magic('capture', '', 'import IPython\nfrom common.functions import get_highlited_function_code, get_features, _concat_row\nfrom common.constatns import data_location, vqa_models_folder, vqa_specs_location #train_data, validation_data, \nfrom common.utils import VerboseTimer\nfrom common.settings import classify_strategy\nfrom common.classes import ClassifyStrategies\nfrom common.model_utils import save_model\nfrom common.os_utils import File')


# #### Loading the Model:

# In[4]:


with VerboseTimer("Loading Model"):
    model = load_model(model_location)


# #### Loading the data:

# In[5]:


logger.debug(f"Loading the data from {data_location}")
with VerboseTimer("Loading Data"):
    with HDFStore(data_location) as store:
        df_data = store['data']  


# In[6]:


vqa_specs = File.load_pickle(vqa_specs_location)
meta_data = vqa_specs.meta_data


# In[7]:


logger.debug(f"df_data Shape: {df_data.shape}")
df_data.head(2)


# ### ATTN: This is for training only on 2 classees: MRI / CT

# In[8]:


# df_data = df_data[(df_data.imaging_device == 'ct')|( df_data.imaging_device == 'mri' )].copy()
# logger.debug(f"df_data Shape: {df_data.shape}")
# np.unique(df_data.imaging_device)


# In[9]:


(df_data.imaging_device).head()


# #### Packaging the data to be in expected input shape

# ##### It makes no sense to train on imageing devices we don't know thier lables

# In[10]:


#ATTN: 
cols_to_remove = ['both', 'unknown']
def filter_out_unknown_devices(df):
    valid_devices = meta_data['img_device_to_ix'].keys()
    return df[df.imaging_device.isin(valid_devices)]


df_data_orig = df_data 
df_data = filter_out_unknown_devices(df_data)


# In[11]:


data_train = df_data[df_data.group == 'train'].copy().reset_index()
data_val = df_data[df_data.group == 'validation'].copy().reset_index()

# print(f'groups:\n{df_data.group.drop_duplicates()}')
# print(len(df_data))
# data_val.head()


# ##### The functions for getting the features & labels:

# In[12]:


from common.functions import get_features, _concat_row
code_get_features = get_highlited_function_code( get_features, remove_comments=True)
code_concat = get_highlited_function_code(_concat_row, remove_comments=True)
IPython.display.display(code_get_features)
IPython.display.display(code_concat)


# #### Defining how to get NLP labels

# In[13]:


def get_nlp_labels():
    labels =  _concat_row(df, 'answer_embedding')
    return labels


# #### Defining how to get Categorial fetaures / labels

# In[14]:


def get_categorial_labels(df, meta):
    lookup_col = 'img_device_to_ix'
    # lookup_col = 'img_device_to_ix'
    ans_to_ix = meta[lookup_col]
    all_classes =  ans_to_ix.keys()
   
    
    data_classes = df['imaging_device']
    class_count = len(all_classes)

    classes_indices = [ans_to_ix[ans] for ans in data_classes if ans in ans_to_ix.keys()]
    categorial_labels = to_categorical(classes_indices, num_classes=class_count)
    
    for i in range(len(categorial_labels)):
        assert np.argmax(categorial_labels[i])== classes_indices[i], 'Expected to get argmax at index of label' 

    return categorial_labels



# with VerboseTimer("Getting categorial validation labels"):
#     categorial_labels_val = get_categorial_labels(df_val, meta_data)
# categorial_labels_train.shape, categorial_labels_val.shape
# del df_train
# del df_val


# In[15]:


meta = meta_data
df = data_train
lookup_col = 'img_device_to_ix'
# lookup_col = 'img_device_to_ix'
ans_to_ix = meta[lookup_col]
all_classes =  ans_to_ix.keys()

data_classes = df['imaging_device']
class_count = len(all_classes)

classes_indices = [ans_to_ix[ans] for ans in data_classes if ans in ans_to_ix.keys()]
categorial_labels = to_categorical(classes_indices, num_classes=class_count)


max(classes_indices),min(classes_indices)
# data_classes
categorial_labels
class_count


# In[16]:


def get_features(df):
    image_features = np.asarray([np.array(im) for im in df['image']])
    question_features = _concat_row(df, 'question_embedding')
    reshaped_q = np.array([a.reshape(a.shape + (1,)) for a in question_features])

    features = ([f for f in [reshaped_q, image_features]])

    return features
def _concat_row(df, col):
    try:    
        return np.concatenate(df[col], axis=0)
    except:        
        File.dump_pickle(fn="aaa.pkl", obj=df[col])
        raise


# In[17]:


if classify_strategy == ClassifyStrategies.CATEGORIAL:    
    p_get_categorial_labels = partial(get_categorial_labels, meta=meta_data)        
    get_labels = p_get_categorial_labels
    
elif classify_strategy == ClassifyStrategies.NLP:   
    get_labels = get_nlp_features_and_labels    

# Note: The shape of answer (for a single recored ) is (number of words, 384)
else:
    raise Exception(f'Unfamilier strategy: {strat}')
print(f'classify stratagy: {classify_strategy}')

with VerboseTimer('Getting train features'):
    features_t = get_features(data_train)   
with VerboseTimer('Getting train labels'):
    labels_t = get_labels(data_train)        
    
with VerboseTimer('Getting train features'):
    features_val = get_features(data_val)
with VerboseTimer('Getting validation labels'):
    labels_val = get_labels(data_val)

# len(features_t[1])


# In[18]:


validation_input = (features_val, labels_val)


# In[19]:


# model.input_layers
# model.input_layers_node_indices
# model.input_layers_tensor_indices
# model.input_mask
# model.input_names


# model.inputs
# model.input
# model.input_spec

# print(f'Wrapper shape:{train_features.shape}')
# model.input_shape, np.concatenate(train_features).shape
# model.input_shape, train_features[0].shape,  train_features[1].shape

print(f'Expectedt shape: {model.input_shape}')
print('---------------------------------------------------------------------------')
print(f'Actual training shape:{features_t[0].shape, features_t[1].shape}')
print(f'Train Labels shape:{labels_t.shape}')
print('---------------------------------------------------------------------------')
print(f'Actual Validation shape:{features_val[0].shape, features_val[1].shape}')
print(f'Validation Labels shape:{labels_val.shape}')


# In[20]:


# from utils.gpu_utils import test_gpu
# test_gpu()


# In[ ]:


from keras.utils import plot_model
# EPOCHS=25
# BATCH_SIZE = 20

EPOCHS= 5
BATCH_SIZE = 30

# train_features = image_name_question
# validation_input = (validation_features, categorial_validation_labels)

## construct the image generator for data augmentation
# aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#                                horizontal_flip=True, fill_mode="nearest")
# train_generator = aug.flow(train_features, categorial_train_labels)

# stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,mode='auto')

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


    

    with VerboseTimer("Training Model"):
#         with get_session() as sess:
#             ktf.set_session(sess)
#             sess.run(tf.global_variables_initializer())
            
        history = model.fit(features_t,labels_t,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=validation_input)
#             sess.close()
            
except Exception as ex:
    logger.error("Got an error training model: {0}".format(ex))
    raise
#     model.summary(print_fn=logger.error)
#     raise
# return model, history


# ### Save trained model:

# In[ ]:


with VerboseTimer("Saving trained Model"):
    name_suffix = f'{classify_strategy}_trained'
    model_fn, summary_fn, fn_image = save_model(model, vqa_models_folder, name_suffix=name_suffix)

msg = f"Summary: {summary_fn}\n"
msg += f"Image: {fn_image}\n"
location_message = f"model_location = '{model_fn}'"


print(msg)
print(location_message)


# In[ ]:


print (location_message.replace('\\','\\\\'))

