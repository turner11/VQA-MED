#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%capture
import IPython
import os
from pandas import HDFStore
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path


# In[2]:


logger = logging.getLogger(__name__)


# In[3]:


from common.settings import get_nlp, data_access
from common.functions import get_highlighted_function_code, get_image,  get_size
from pre_processing.prepare_data import get_text_features, pre_process_raw_data
from common.utils import VerboseTimer


# ### Preparing the data for training

# #### Getting the nlp engine

# In[4]:


nlp = get_nlp()


# #### Where get_nlp is defined as:

# In[5]:


code = get_highlighted_function_code(get_nlp,remove_comments=True)
IPython.display.display(code)


# In[6]:


image_name_question = data_access.load_raw_input()


# In[7]:


image_name_question.head()


# ##### This is just for performance and quick debug cycles! remove before actual trainining:

# ### Aditional functions we will use:

# #### get_text_features:

# In[8]:


code = get_highlighted_function_code(get_text_features,remove_comments=True)
IPython.display.display(code)


# #### get_image:

# In[9]:


code = get_highlighted_function_code(get_image,remove_comments=True)
IPython.display.display(code)


# #### pre_process_raw_data:

# In[10]:


code = get_highlighted_function_code(pre_process_raw_data,remove_comments=True)
IPython.display.display(code)


# ### Clean and enrich the data

# In[11]:


from pre_processing.data_enrichment import enrich_data
from pre_processing.data_cleaning import clean_data

orig_image_name_question = image_name_question.copy()
image_name_question = clean_data(image_name_question)
image_name_question = enrich_data(image_name_question)


# In[12]:


groups = image_name_question.groupby('group')
groups.describe()


# In[13]:


image_name_question.head()
image_name_question.sample(n=7)


# ### Do the actual pre processing

# #### If running in an exported notebook, use the following:
# (indent everything to be under the main guard) - for avoiding recursive spawning of processes

# In[14]:


from multiprocessing import freeze_support
if __name__ == '__main__':
    print('in main')
    freeze_support()


# Note:  
# This might take a while...

# In[15]:


logger.debug('----===== Preproceccing train data =====----')
image_name_question_processed = pre_process_raw_data(image_name_question)


# In[16]:


image_name_question_processed.sample(5)


# In[17]:


image_name_question_processed[image_name_question_processed.image_name == 'synpic52143.jpg'].head()


# #### Saving the data, so later on we don't need to compute it again

# In[18]:


saved_path = data_access.save_processed_data(image_name_question_processed)


# In[19]:


print(f'Data saved at:\n{saved_path}')

