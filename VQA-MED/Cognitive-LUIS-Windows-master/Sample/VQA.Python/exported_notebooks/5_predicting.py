
# coding: utf-8

# In[1]:


#From Step #2:
vqa_specs_location = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\data\\vqa_specs.pkl'


# In[2]:


model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180629_1220_23\\vqa_model_ClassifyStrategies.NLP_trained.h5'


# In[3]:


# %%capture
import os
import numpy as np
from pandas import HDFStore
from vqa_logger import logger 
from enum import Enum
import IPython
from keras.models import load_model


# In[4]:


from common.constatns import images_path_test
from common.utils import VerboseTimer
from parsers.VQA18 import Vqa18Base
from common.functions import get_size, get_highlited_function_code, normalize_data_strucrture
from vqa_logger import logger
from common.os_utils import File 


# In[5]:


with VerboseTimer("Loading Model"):
    model = load_model(model_location)


# In[6]:


vqa_specs = File.load_pickle(vqa_specs_location)
data_location = vqa_specs.data_location
data_location


# In[7]:


code = get_highlited_function_code(normalize_data_strucrture,remove_comments=True)
IPython.display.display(code)


# In[8]:


logger.debug(f"Loading test data from {data_location}")
with VerboseTimer("Loading Test Data"):
    with HDFStore(data_location) as store:        
        df_data = store['test']


# In[9]:


df_data.head()


# In[11]:


#TODO: Duplicate:


# In[21]:


def concate_row(df, col):
    return np.concatenate(df[col], axis=0)

def get_features_and_labels(df):
    image_features = np.asarray([np.array(im) for im in df['image']])
    # np.concatenate(image_features['question_embedding'], axis=0).shape
    question_features = concate_row(df, 'question_embedding') 

    features = ([f for f in [question_features, image_features]])
    labels = None# concate_row(df, 'answer_embedding')
    return features, labels


# In[22]:


features, _ = get_features_and_labels(df_data)


# In[25]:



p = model.predict(features)


# In[26]:


p

