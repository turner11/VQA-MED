
# coding: utf-8

# In[2]:


# %%capture
import IPython
import os
import numpy as np
import pandas as pd
from pandas import HDFStore
import spacy
from keras.utils import to_categorical
import cv2
from collections import defaultdict

from vqa_logger import logger
from common.os_utils import File


# In[3]:


from common.constatns import train_data, validation_data, data_location, raw_data_location
from common.settings import input_length, embedding_dim, image_size, seq_length, get_nlp
from common.functions import get_highlited_function_code, get_image, get_text_features, pre_process_raw_data, get_size
from common.utils import VerboseTimer


# ### Preparing the data for training

# #### Getting the nlp engine

# In[4]:


nlp = get_nlp()


# #### Where get_nlp is defined as:

# In[5]:


code = get_highlited_function_code(get_nlp,remove_comments=True)
IPython.display.display(code)


# In[6]:


with HDFStore(raw_data_location) as store:
    image_name_question = store['data']
# df_train = image_name_question[image_name_question.group == 'train']
# df_val = image_name_question[image_name_question.group == 'validation']

# from parsers.VQA18 import Vqa18Base
# df_train = Vqa18Base.get_instance(train_data.processed_xls).data            
# df_val = Vqa18Base.get_instance(validation_data.processed_xls).data


# ##### This is just for performance and quick debug cycles! remove before actual trainining:

# In[7]:


# image_name_question = image_name_question.head(5)
# image_name_question_val = image_name_question_val.head(5)


# ### Aditional functions we will use:

# #### get_text_features:

# In[8]:


code = get_highlited_function_code(get_text_features,remove_comments=True)
IPython.display.display(code)


# #### get_image:

# In[9]:


code = get_highlited_function_code(get_image,remove_comments=True)
IPython.display.display(code)


# #### pre_process_raw_data:

# In[10]:


code = get_highlited_function_code(pre_process_raw_data,remove_comments=True)
IPython.display.display(code)


# ### Clean and enrich the data

# In[11]:


from common.functions import enrich_data, clean_data
image_name_question = clean_data(image_name_question)
image_name_question = enrich_data(image_name_question)


# In[12]:


image_name_question.head()


# In[13]:


image_name_question.groupby('group').describe()
image_name_question[['imaging_device','image_name']].groupby('imaging_device').describe()


# ### Do the actual pre processing
# Note:  
# This might take a while...

# In[14]:


# # # # RRR
# # # logger.debug('Getting answers embedding')
# df = image_name_question
# df['l'] = df.answer.apply(lambda a: len(str(a)))
# df[df.l > 2].sort_values('l')
# # print(len(df[(df.answer == np.nan) | (df.question == np.nan)]))


# # df['answer'].apply(lambda q: get_text_features(q))
# # # a= df['answer'].apply(lambda q: 0 if q == np.nan else 1)
# # # sum(a), len(a), len(image_name_question)

# import json
# # json.load(open)
# a = df[df.group == 'test']['answer'].values[0]
# type(a)



# In[15]:


logger.debug('----===== Preproceccing train data =====----')
image_locations = train_data.images_path
with VerboseTimer("Pre processing training data"):
    image_name_question_processed = pre_process_raw_data(image_name_question)


# In[16]:


# logger.debug('----===== Preproceccing validation data =====----')
# image_locations = validation_data.images_path
# with VerboseTimer("Pre processing validation data"):
#     image_name_question_val = pre_process_raw_data(image_name_question_val, image_locations)


# #### Saving the data, so later on we don't need to compute it again

# In[19]:


logger.debug("Saving the data")

item_to_save = image_name_question_processed
# item_to_save = image_name_question.head(10)

# remove if exists
try:
    os.remove(data_location)
except OSError:
    pass


with VerboseTimer("Saving model training data"):
    with HDFStore(data_location) as store:
        store['data']  = image_name_question_processed[(image_name_question_processed.group == 'train') | (image_name_question_processed.group == 'validation')]
        store['test']  = image_name_question_processed[image_name_question_processed.group == 'test']
        
size = get_size(data_location)
logger.debug(f"training data's file size was: {size}")


# In[20]:


print('Data saved at:')
f'{data_location}'

