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


import vqa_logger 
logger = logging.getLogger(__name__)


# In[3]:


from common.constatns import raw_data_location, data_location
from common.settings import get_nlp
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


raw_data_location


# In[7]:


with HDFStore(raw_data_location) as store:
    image_name_question = store['data']


# In[8]:


image_name_question.head()


# ##### This is just for performance and quick debug cycles! remove before actual trainining:

# In[9]:


# image_name_question = image_name_question.head(5)
# image_name_question_val = image_name_question_val.head(5)


# ### Aditional functions we will use:

# #### get_text_features:

# In[10]:


code = get_highlighted_function_code(get_text_features,remove_comments=True)
IPython.display.display(code)


# #### get_image:

# In[11]:


code = get_highlighted_function_code(get_image,remove_comments=True)
IPython.display.display(code)


# #### pre_process_raw_data:

# In[12]:


code = get_highlighted_function_code(pre_process_raw_data,remove_comments=True)
IPython.display.display(code)


# ### Clean and enrich the data

# In[13]:


from pre_processing.data_enrichment import enrich_data
from pre_processing.data_cleaning import clean_data

orig_image_name_question = image_name_question.copy()
image_name_question = clean_data(image_name_question)
image_name_question = enrich_data(image_name_question)


# In[14]:


groups = image_name_question.groupby('group')
groups.describe()
image_name_question[['imaging_device','image_name']].groupby('imaging_device').describe()


# ### Do the actual pre processing

# #### If running in an exported notebook, use the following:
# (indent everything to be under the main guard) - for avoiding recursive spawning of processes

# In[15]:


from multiprocessing import freeze_support
if __name__ == '__main__':
    print('in main')
    freeze_support()


# Note:  
# This might take a while...

# In[16]:


logger.debug('----===== Preproceccing train data =====----')
with VerboseTimer("Pre processing training data"):
    image_name_question_processed = pre_process_raw_data(image_name_question)


# In[17]:


image_name_question_processed.head()


# In[18]:


image_name_question[image_name_question.image_name == 'synpic52143.jpg'].head()


# #### Saving the data, so later on we don't need to compute it again

# ### TODO: need to add question classification taking in consideration 2019 data

# In[19]:


def add_dataframe_to_data_set(df, location):
    table = pa.Table.from_pandas(df)

    pq.write_to_dataset(
        table,
        root_path=str(location),#'output.parquet',
        partition_cols=['group'],
    )
#train_df.to_parquet(fname='',engine='pyarrow',partition_cols=)


# In[20]:


logger.debug("Saving the data")
item_to_save = image_name_question_processed
# item_to_save = image_name_question.head(10)

# remove if exists
try:
    os.remove(data_location)
except OSError:
    pass


train_df = image_name_question_processed[(image_name_question_processed.group == 'train') | (image_name_question_processed.group == 'validation')]
test_df = image_name_question_processed[image_name_question_processed.group == 'test']
light = image_name_question_processed[['image_name', 'question', 'answer', 'group', 'path', 'imaging_device']]




root = Path(data_location)
with VerboseTimer("Saving model training data"):
    add_dataframe_to_data_set(image_name_question_processed, root)
#     light.to_hdf(data_location, 'light', mode='w', data_columns=['image_name', 'imaging_device', 'path'], format='table')    
#     add_dataframe_to_data_set(train_df, root/'train')
#     add_dataframe_to_data_set(test_df, root/'test')
        
size = get_size(data_location)
logger.debug(f"training data's file size was: {size}")


# In[21]:


print('Data saved at:')
f'{data_location}'

