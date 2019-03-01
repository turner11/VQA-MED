#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from classes.vqa_model_predictor import VqaModelPredictor, DefaultVqaModelPredictor
from common.DAL import get_models_data_frame, get_model
from common.DAL import ModelScore
from common import DAL

from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.functions import get_highlighted_function_code
import vqa_logger 


# In[3]:


df_models = get_models_data_frame()
try:
    df_show = df_models.sort_values(by=['wbss', 'bleu'], ascending=False).head()
except KeyError: #if no scode yet
    df_show = df_models
    df_show['wbss'] = np.nan
    df_show['bleu'] = np.nan

    
df_show.tail()


# In[4]:


import logging
import  vqa_logger 
logger = logging.getLogger(__name__)
import IPython


# In[5]:


model_id = None#int(model_id)
mp = DefaultVqaModelPredictor(model_id)
mp


# In[6]:


mp.df_validation.head(2)


# In[7]:


code = get_highlighted_function_code(mp.predict,remove_comments=False)
IPython.display.display(code)


# In[8]:


df_data = mp.df_validation
df_predictions = mp.predict(mp.df_validation)
df_predictions.head()


# In[9]:


df_predictions.describe()


# #### Take a look at results for a single image:

# In[85]:



image_name = df_predictions.image_name.sample(1).values[0]

df_image = df_predictions[df_predictions.image_name == image_name]
# print(f'Result: {set(df_image.prediction)}')

image_path = df_image.path.values[0]
df_image


# In[86]:


from IPython.display import Image, HTML, display_html
image = Image(filename = image_path, width=400, height=400)
image


# ## Evaluating the Model

# In[13]:


validation_prediction = df_predictions
predictions = validation_prediction.prediction.values
ground_truth = validation_prediction.answer.values
results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
print(f'Got results of\n{results}')


# ##### Add the core to DB:

# In[14]:


model_db_id = mp.model_idx_in_db
assert model_db_id >= 0 
model_db_id


# In[15]:


bleu = results['bleu']
wbss = results['wbss']
model_score = ModelScore(model_db_id, bleu=bleu, wbss=wbss)
model_score


# In[16]:


DAL.insert_dal(model_score)

