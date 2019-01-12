
# coding: utf-8

# In[1]:


import os
import pandas as pd
from pandas import HDFStore
from nltk.corpus import stopwords
import IPython


# In[10]:


from common.functions import get_highlighted_function_code
from common.constatns import data_location, vqa_specs_location, fn_meta
from common.settings import embedding_dim, seq_length
from common.classes import VqaSpecs
from common.utils import VerboseTimer
from common.os_utils import File
from pre_processing.meta_data import create_meta


# ### Preprocessing and creating meta data

# Get the data itself, Note the only things required in dataframe are:
# 1. image_name
# 2. question
# 3. answer
# 

# In[4]:


print(f'loading from:\n{data_location}')
with VerboseTimer("Loading Data"):
    with HDFStore(data_location) as store:
         df_data = store['data']
        
df_data = df_data[df_data.group.isin(['train','validation'])]
print(f'Data length: {len(df_data)}')        
df_data.head(2)


# In[5]:


import numpy as np
d = df_data[df_data.imaging_device.isin(['ct','mri'])]
print(np.unique(df_data.imaging_device))
print(np.unique(d.imaging_device))


# #### We will use this function for creating meta data:

# In[6]:


code = get_highlighted_function_code(create_meta, remove_comments=False)
IPython.display.display(code)  


# In[7]:


print("----- Creating meta -----")
meta_data = create_meta(df_data, fn_meta)

with HDFStore(fn_meta) as metadata_store:           
    df_words = metadata_store['words']
    df_answers = metadata_store['answers']
    df_imaging_device = metadata_store['imaging_devices']
    
df_words.head()


# #### Saving the data, so later on we don't need to compute it again

# In[8]:


def get_vqa_specs(meta_location):    
    dim = embedding_dim
    s_length = seq_length    
    return VqaSpecs(embedding_dim=dim, 
                    seq_length=s_length, 
                    data_location=os.path.abspath(data_location),
                    meta_data_location=os.path.abspath(meta_location))

vqa_specs = get_vqa_specs(fn_meta)

# Show waht we got...
vqa_specs


# In[12]:


File.dump_pickle(vqa_specs, vqa_specs_location)
print(f"VQA Specs saved to:\n{vqa_specs_location}")


# ##### Test Loading:

# In[13]:


loaded_vqa_specs = File.load_pickle(vqa_specs_location)
loaded_vqa_specs


# In[14]:


print (f"vqa_specs_location = '{vqa_specs_location}'".replace('\\','\\\\'))

