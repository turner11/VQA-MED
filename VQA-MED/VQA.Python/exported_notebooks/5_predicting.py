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


# In[7]:


mp.df_validation.head(2)


# In[8]:


code = get_highlighted_function_code(mp.predict,remove_comments=False)
IPython.display.display(code)


# In[9]:


df_data = mp.df_validation
df_predictions = mp.predict(mp.df_validation)
df_predictions.head()


# In[10]:


df_predictions.describe()


# #### Take a look at results for a single image:

# In[ ]:



from IPython.display import Image, HTML

df = pd.DataFrame(['./image01.png', './image02.png'], columns = ['Image'])

def path_to_image_html(path):
    return '<img src="'+ path + '"/>'

pd.set_option('display.max_colwidth', -1)

HTML(df.to_html(escape=False ,formatters=dict(Image=path_to_image_html)))


# In[12]:



image_name = df_predictions.image_name.sample(1).values[0]

df_image = df_predictions[df_predictions.image_name == image_name]
# print(f'Result: {set(df_image.prediction)}')

image_path = df_image.path.values[0]
df_image


# In[13]:


from IPython.display import Image
Image(filename = image_path, width=400, height=400)


# ## Evaluating the Model

# In[14]:


validation_prediction = df_predictions
predictions = validation_prediction.prediction.values
ground_truth = validation_prediction.answer.values
results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
print(f'Got results of\n{results}')


# ##### Add the core to DB:

# In[20]:


model_db_id = mp.model_idx_in_db
assert model_db_id >= 0 
model_db_id


# In[24]:


bleu = results['bleu']
wbss = results['wbss']
model_score = ModelScore(model_db_id, bleu=bleu, wbss=wbss)
model_score


# In[25]:


DAL.insert_dal(model_score)

