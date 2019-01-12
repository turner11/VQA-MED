
# coding: utf-8

# # Bringing data to expected format

# ### This may change according to the format of data you have...

# In[1]:


import os
from pandas import HDFStore
import pandas as pd
import IPython

from common.constatns import train_data, validation_data, test_data, raw_data_location, images_folder_train, images_folder_validation, images_path_test
from common.utils import VerboseTimer
from parsers.VQA18 import Vqa18Base
from common.functions import get_size, get_highlighted_function_code, normalize_data_structure
import logging
logger = logging.getLogger(__name__)


# In[2]:


# TODO: Change this to use the original format from image_clef
df_train = Vqa18Base.get_instance(train_data.processed_xls).data    
df_valid = Vqa18Base.get_instance(validation_data.processed_xls).data
df_test = Vqa18Base.get_instance(test_data.processed_xls).data


# ### For bringing the data to a normalized state we will use the function 'normalize_data_strucrture'
# Defined as:

# In[3]:


code = get_highlighted_function_code(normalize_data_structure, remove_comments=True)
IPython.display.display(code)


# In[4]:


df_t = normalize_data_structure(df_train, 'train', images_folder_train)
df_v = normalize_data_structure(df_valid, 'validation', images_folder_validation)
df_test = normalize_data_structure(df_test, 'test', images_path_test)


df = pd.concat([df_t, df_v, df_test])  # .reset_index()
#         folder = images_folder_train if group == 'train' else images_folder_validation
    
df.describe()


# ### Save the data

# In[5]:


# remove if exists
try:
    os.remove(raw_data_location)
except OSError:
    pass

with VerboseTimer("Saving raw data"):
    with HDFStore(raw_data_location) as store:
        store['data']  = df
      
        

size = get_size(raw_data_location)
logger.debug(f"raw data's file size was: {size}")

