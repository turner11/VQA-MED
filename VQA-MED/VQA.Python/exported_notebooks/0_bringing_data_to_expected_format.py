#!/usr/bin/env python
# coding: utf-8

# # Bringing data to expected format

# ### This may change according to the format of data you have...

# In[1]:


import os
from pandas import HDFStore
import pandas as pd
import IPython

from common.settings import train_data, validation_data, test_data, data_access

from common.functions import get_size, get_highlighted_function_code
from pre_processing.prepare_data import normalize_data_strucrture
from parsers.data_loader import DataLoader
import vqa_logger 
import logging

logger = logging.getLogger(__name__)


# In[3]:


df_train = DataLoader.get_data(train_data.qa_path)
df_valid = DataLoader.get_data(validation_data.qa_path)
df_test = DataLoader.get_data(test_data.qa_path)


# ### For bringing the data to a normalized state we will use the function 'normalize_data_strucrture'
# Defined as:

# In[4]:


code = get_highlighted_function_code(normalize_data_strucrture,remove_comments=True)
IPython.display.display(code)


# In[7]:


pd.concat([df_train.head(),df_test.head()])


# In[9]:


def normalize_data(df, set_info):
    normed = normalize_data_strucrture(df, set_info.tag, set_info.images_folder)
    return normed

df_nt = normalize_data(df_train, train_data)
df_nv = normalize_data(df_valid, validation_data)
df_ntest = normalize_data(df_test, test_data)

df = pd.concat([df_nt, df_nv, df_ntest])  # .reset_index()
#         folder = images_folder_train if group == 'train' else images_folder_validation
    
df.describe()


# ### Save the data

# In[10]:


save_location = data_access.save_raw_input(df)


# In[11]:


print(f'data saved to:\n{save_location}')

