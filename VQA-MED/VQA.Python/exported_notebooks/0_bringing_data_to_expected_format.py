#!/usr/bin/env python
# coding: utf-8

# In[2]:


import IPython
from common.functions import get_size, get_highlighted_function_code


# # Bringing data to expected format

# The raw input data is a pipe-delimited text in the following format:  
# *image_id | question | answer*  
# Before we start the process, we will bring the data into a more convinient format - Pandas DataFrame  

# For bringing the data to a normalized state we will use the function **normalize_data_strucrture**, Defined as depiceted below.  
# The process will be done for train, validation and test sets, and finally, combined to gether to a single dataframe.

# In[3]:


from pre_processing.prepare_data import normalize_data_strucrture
code = get_highlighted_function_code(normalize_data_strucrture,remove_comments=True)
IPython.display.display(code)


# ---
# ## The code:

# In[4]:


import os
from pandas import HDFStore
import pandas as pd
from common.settings import train_data, validation_data, test_data, data_access
from parsers.data_loader import DataLoader
import vqa_logger 
import logging

logger = logging.getLogger(__name__)


# In[5]:


df_train = DataLoader.get_data(train_data.qa_path)
df_valid = DataLoader.get_data(validation_data.qa_path)
df_test = DataLoader.get_data(test_data.qa_path)


# In[8]:


def normalize_data(df, set_info):
    normed = normalize_data_strucrture(df, set_info.tag, set_info.images_folder)
    return normed

df_nt = normalize_data(df_train, train_data)
df_nv = normalize_data(df_valid, validation_data)
df_ntest = normalize_data(df_test, test_data)

df = pd.concat([df_nt, df_nv, df_ntest])
    
df.describe()
df.head()


# ### Save the data

# In[10]:


save_location = data_access.save_raw_input(df)


# In[11]:


print(f'data saved to:\n{save_location}')

