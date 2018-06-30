
# coding: utf-8

# # Bringing data to expected format

# ### This may change according to the format of data you have...

# In[1]:


import os
from pandas import HDFStore
import pandas as pd

from common.constatns import train_data, validation_data,  raw_data_location, images_folder_train, images_folder_validation 
from common.utils import VerboseTimer
from parsers.VQA18 import Vqa18Base
from common.functions import get_size
from vqa_logger import logger


# In[2]:


# TODO: Change this to use the original format from image_clef
df_train = Vqa18Base.get_instance(train_data.processed_xls).data    
df_valid = Vqa18Base.get_instance(validation_data.processed_xls).data 

cols = ['image_name', 'question', 'answer']
df_t = df_train[cols].copy()
df_v = df_valid[cols].copy()

df_t['group'] = 'train'
df_v['group'] = 'validation'


df = pd.concat([df_t, df_v])#.reset_index()

def get_image_path(group, image_name):
    assert group in ['train', 'validation']
    folder = images_folder_train if group == 'train' else images_folder_validation    
    return os.path.join(folder, image_name)


df['image_name'] = df['image_name'].apply(lambda q: q if q.lower().endswith('.jpg') else q + '.jpg')
df['path'] = df.apply(lambda x:  get_image_path(x['group'],x['image_name']),axis=1) #x: get_image_path(x['group'],x['image_name'])

df.describe()


# ### Save the data

# In[3]:


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


# In[9]:


# df.reset_index()
# df.head()
a = df[df.group == 'validation'].path[0]
os.path.exists(a)

