
# coding: utf-8

# In[1]:


# %%capture
import IPython
import os

from pandas import HDFStore


import logging



logger = logging.getLogger(__name__)



# In[2]:


from common.constatns import train_data, data_location, raw_data_location
from common.settings import get_nlp
from common.functions import get_highlighted_function_code, get_image, get_size
from pre_processing.prepare_data import get_text_features, pre_process_raw_data
from common.utils import VerboseTimer


# ### Preparing the data for training

# #### Getting the nlp engine

# In[3]:


nlp = get_nlp()


# #### Where get_nlp is defined as:

# In[4]:


code = get_highlighted_function_code(get_nlp, remove_comments=True)
IPython.display.display(code)


# In[5]:


with HDFStore(raw_data_location) as store:
    image_name_question = store['data']


# In[6]:


image_name_question.head()


# ##### This is just for performance and quick debug cycles! remove before actual trainining:

# In[7]:


# image_name_question = image_name_question.head(5)
# image_name_question_val = image_name_question_val.head(5)


# ### Aditional functions we will use:

# #### get_text_features:

# In[8]:


code = get_highlighted_function_code(get_text_features, remove_comments=True)
IPython.display.display(code)


# #### get_image:

# In[9]:


code = get_highlighted_function_code(get_image, remove_comments=True)
IPython.display.display(code)


# #### pre_process_raw_data:

# In[10]:


code = get_highlighted_function_code(pre_process_raw_data, remove_comments=True)
IPython.display.display(code)


# ### Clean and enrich the data

# In[11]:

from pre_processing.data_enrichment import enrich_data
from pre_processing.data_cleaning import clean_data
orig_image_name_question = image_name_question.copy()
image_name_question = clean_data(image_name_question)
image_name_question = enrich_data(image_name_question)


# In[12]:


image_name_question[image_name_question.image_name == '0392-100X-33-350-g002.jpg'].head()
image_name_question.head()


# In[13]:


image_name_question.groupby('group').describe()
image_name_question[['imaging_device','image_name']].groupby('imaging_device').describe()


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
    image_locations = train_data.images_path
    with VerboseTimer("Pre processing training data"):
        image_name_question_processed = pre_process_raw_data(image_name_question)


    # In[16]:


    image_name_question_processed.head()


    # #### Saving the data, so later on we don't need to compute it again

    # In[17]:


    image_name_question_processed.imaging_device.drop_duplicates()


    # In[17]:


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
    light = image_name_question_processed[['image_name', 'question', 'answer', 'group', 'path', 'tumor', 'hematoma', 'brain', 'abdomen', 'neck', 'liver', 'imaging_device']]


    with VerboseTimer("Saving model training data"):
        light.to_hdf(data_location, 'light', mode='w', data_columns=['image_name', 'imaging_device', 'path'], format='table')
        with HDFStore(data_location) as store:
            store['data']  = train_df
            store['test']  = test_df

    size = get_size(data_location)
    logger.debug(f"training data's file size was: {size}")


    # In[18]:


    data_location


    # In[19]:


    # import numpy as np
    # d = train_df[train_df.imaging_device.isin(['ct','mri'])]
    # print(np.unique(train_df.imaging_device))
    # print(np.unique(d.imaging_device))


    # In[20]:


    print('Data saved at:')
    f'{data_location}'

