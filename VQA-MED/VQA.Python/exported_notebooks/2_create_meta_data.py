#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import pandas as pd
from pandas import HDFStore
from nltk.corpus import stopwords
import IPython


# In[41]:


from common.functions import get_highlighted_function_code
from common.settings import embedding_dim, seq_length, data_access
from common.classes import VqaSpecs
from common.utils import VerboseTimer
from common.os_utils import File
from pre_processing.meta_data import create_meta
pd.set_option('display.max_colwidth', -1)


# ### Preprocessing and creating meta data

# Get the data itself, Note the only things required in dataframe are:
# 1. image_name
# 2. processed question
# 3. processed answer
# 

# In[42]:


# index	image_name	question	answer	group	path	original_question	original_answer	tumor	hematoma	brain	abdomen	neck	liver	imaging_device	answer_embedding	question_embedding	is_imaging_device_question
df_data = data_access.load_processed_data(columns=['path','question','answer', 'processed_question','processed_answer', 'group','question_category'])        
df_data = df_data[df_data.group.isin(['train','validation'])]
print(f'Data length: {len(df_data)}')        
df_data.sample(2)


# #### We will use this function for creating meta data:

# In[43]:


code = get_highlighted_function_code(create_meta,remove_comments=False)
IPython.display.display(code)  


# In[44]:


print("----- Creating meta -----")
meta_data_dict = create_meta(df_data)


# #### Saving the data, so later on we don't need to compute it again

# In[45]:


print("----- Saving meta -----")
data_access.save_meta(meta_data_dict)


# ##### Test Loading:

# In[51]:


loaded_meta = data_access.load_meta()
answers_meta = loaded_meta['answers']
words_meta = loaded_meta['words']

answers_meta.sample(5)
# words_meta.sample(5)

