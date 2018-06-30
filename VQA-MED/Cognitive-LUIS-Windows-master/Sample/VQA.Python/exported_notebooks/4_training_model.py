
# coding: utf-8

# ### Training the model

# In[1]:


model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180629_1514_58\\vqa_model_NLP.h5'
strategy_str = 'NLP'


# ### Preparing the data for training

# In[2]:


# %%capture
import os
import numpy as np
from pandas import HDFStore
from vqa_logger import logger 
from enum import Enum
from keras.models import load_model


# In[3]:


from common.constatns import data_location, vqa_models_folder #train_data, validation_data, 
from common.utils import VerboseTimer
from common.settings import classify_strategy
from common.classes import ClassifyStrategies
from common.model_utils import save_model


# #### Loading the Model:

# In[4]:


with VerboseTimer("Loading Model"):
    model = load_model(model_location)


# #### Loading the data:

# In[13]:


logger.debug(f"Loading the data from {data_location}")
with VerboseTimer("Loading Data"):
    with HDFStore(data_location) as store:
        df_data = store['data']  


# In[14]:


logger.debug(f"df_data Shape: {df_data.shape}")
df_data.head(2)


# #### Packaging the data to be in expected input shape

# In[16]:


data_train = df_data[df_data.group == 'train']
data_val = df_data[df_data.group == 'validation']


# print(f'groups:\n{df_data.group.drop_duplicates()}')
# print(len(df_data))
# data_val.head()


# In[18]:


def concate_row(df, col):
    return np.concatenate(df[col], axis=0)

def get_features_and_labels(df):
    image_features = np.asarray([np.array(im) for im in df['image']])
    # np.concatenate(image_features['question_embedding'], axis=0).shape
    question_features = concate_row(df, 'question_embedding') 

    features = ([f for f in [question_features, image_features]])
    labels =  concate_row(df, 'answer_embedding')
    return features, labels



logger.debug('Getting train features')
features_t, labels_t = get_features_and_labels(data_train)
logger.debug('Getting validation features')
features_val, labels_val = get_features_and_labels(data_val)

# Note: The shape of answer (for a single recored ) is (number of words, 384)


# An attempt for using categorial classes:

# In[19]:


if classify_strategy == ClassifyStrategies.CATEGORIAL:
    labels_t = categorial_labels_train
    labels_val = categorial_labels_val    
elif classify_strategy == ClassifyStrategies.NLP:
    pass
else:
    raise Exception(f'Unfamilier strategy: {strat}')
classify_strategy

len(features_t[1])


# In[20]:


validation_input = (features_val, labels_val)


# In[21]:


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


# In[ ]:


from keras.utils import plot_model
EPOCHS=25
BATCH_SIZE = 20

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
    with VerboseTimer("Training Model"):
        history = model.fit(features_t,labels_t,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=validation_input)
except Exception as ex:
    logger.error("Got an error training model: {0}".format(ex))
#     model.summary(print_fn=logger.error)
    raise
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

