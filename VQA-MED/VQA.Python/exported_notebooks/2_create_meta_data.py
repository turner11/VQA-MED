#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
from pandas import HDFStore
from nltk.corpus import stopwords
import IPython


# In[6]:


from common.functions import get_highlighted_function_code
from common.settings import embedding_dim, seq_length, data_access
from common.classes import VqaSpecs
from common.utils import VerboseTimer
from common.os_utils import File
from pre_processing.meta_data import create_meta


# In[7]:


vqa_specs_location = data_access.vqa_specs_location
fn_meta =  data_access.fn_meta


# ### Preprocessing and creating meta data

# Get the data itself, Note the only things required in dataframe are:
# 1. image_name
# 2. processed question
# 3. processed answer
# 

# In[8]:


# index	image_name	question	answer	group	path	original_question	original_answer	tumor	hematoma	brain	abdomen	neck	liver	imaging_device	answer_embedding	question_embedding	is_imaging_device_question
df_data = data_access.load_processed_data(columns=['path','question','answer', 'processed_question','processed_answer', 'group','question_category'])        
df_data = df_data[df_data.group.isin(['train','validation'])]
print(f'Data length: {len(df_data)}')        
df_data.sample(2)


# #### We will use this function for creating meta data:

# In[9]:


code = get_highlighted_function_code(create_meta,remove_comments=False)
IPython.display.display(code)  


# In[11]:


print("----- Creating meta -----")
meta_data_dict = create_meta(df_data)
meta_data_dict


# #### Saving the data, so later on we don't need to compute it again

# In[ ]:


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


# In[ ]:


File.dump_pickle(vqa_specs, vqa_specs_location)
print(f"VQA Specs saved to:\n{vqa_specs_location}")


# ##### Test Loading:

# In[ ]:


loaded_vqa_specs = File.load_pickle(vqa_specs_location)
loaded_vqa_specs


# In[ ]:


print (f"vqa_specs_location = '{vqa_specs_location}'".replace('\\','\\\\'))

