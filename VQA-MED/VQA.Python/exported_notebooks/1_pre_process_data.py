#!/usr/bin/env python
# coding: utf-8

# # Preprocessing Data

# In this notebook we will produce the  model input data.  
# For that, we will  **clean** the data and **enrich**  it, and then **extract** features from raw data.  
# This includes:  
# 1. Cleaning the questions / answers (removing stop words, tokenizing)
# 2. Enrichment: Marking diagnosis using thumb rules (Note: Eventually, we did not use this data)  
# 3. Enrichment: Adding a question category to data (given in train / validation sets, thumb rules + prediction to test set)
# 4. Pre processing: Getting Embedding for questions (get_text_features)  
#     For this, we used Spacy's NLP package

# ### Some main functions we used:

# In[1]:


import IPython
from common.functions import get_highlighted_function_code


# #### get_text_features for getting embedding of text

# In[2]:


from pre_processing.prepare_data import get_text_features
code = get_highlighted_function_code(get_text_features,remove_comments=True)
IPython.display.display(code)


# #### pre_process_raw_data for the data pre processing:

# In[3]:


from pre_processing.prepare_data import  pre_process_raw_data
code = get_highlighted_function_code(pre_process_raw_data,remove_comments=True)
IPython.display.display(code)


# #### Cleaning the data:

# In[7]:


# from pre_processing.data_cleaning import clean_data
# code = get_highlighted_function_code(clean_data,remove_comments=True)
# IPython.display.display(code)


# #### Enriching the data

# In[6]:


# from pre_processing.data_enrichment import enrich_data
# code = get_highlighted_function_code(enrich_data,remove_comments=True)
# IPython.display.display(code)


# ---
# ## The code:

# In[8]:


# %%capture
from common.settings import get_nlp, data_access
from common.functions import get_image,  get_size
from pre_processing.prepare_data import get_text_features, pre_process_raw_data
from pre_processing.data_enrichment import enrich_data
from pre_processing.data_cleaning import clean_data
from common.utils import VerboseTimer
from collections import Counter
import os
from pandas import HDFStore
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path


# In[9]:


logger = logging.getLogger(__name__)


# ##### Getting the nlp engine
# (doing it once - it is a singleton)

# In[10]:


nlp = get_nlp()


# Getting the raw input

# In[11]:


image_name_question = data_access.load_raw_input()


# In[12]:


image_name_question.head()


# ### Clean and enrich the data

# In[13]:


orig_image_name_question = image_name_question.copy()
image_name_question = clean_data(image_name_question)
image_name_question = enrich_data(image_name_question)


# In[14]:


groups = image_name_question.groupby('group')
groups.describe()


# In[15]:


image_name_question.head()
image_name_question.sample(n=4)


# ## Do the actual pre processing

# #### If running in an exported notebook, use the following:
# (indent everything to be under the main guard) - for avoiding recursive spawning of processes

# In[14]:


from multiprocessing import freeze_support
if __name__ == '__main__':
    print('in main')
    freeze_support()


# Note:  
# This might take a while...

# In[17]:


logger.debug('----===== Preproceccing train data =====----')
image_name_question_processed = pre_process_raw_data(image_name_question)


# In[18]:


image_name_question_processed.sample(2)


# Take a look at data of a single image:

# In[17]:


image_name_question_processed[image_name_question_processed.image_name == 'synpic52143.jpg'].head()


# In[19]:


from collections import Counter


# How many categories did we get for questions?

# In[21]:


print('--Test--')
print(Counter(image_name_question_processed[image_name_question_processed.group=='test'].question_category.values))
print('--All--')
print(Counter(image_name_question_processed.question_category.values))


# #### Saving the data, so later on we don't need to compute it again

# In[18]:


saved_path = data_access.save_processed_data(image_name_question_processed)


# In[19]:


print(f'Data saved at:\n{saved_path}')

