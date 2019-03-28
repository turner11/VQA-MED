#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from classes.vqa_model_predictor import VqaModelPredictor, DefaultVqaModelPredictor
from data_access.api import DataAccess, SpecificDataAccess
from common.DAL import get_models_data_frame, get_model
from common.DAL import ModelScore
from common.settings import data_access as data_access_api
from common import DAL


from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.functions import get_highlighted_function_code
import vqa_logger 


# In[3]:


from keras import backend as keras_backend
keras_backend.clear_session()


# In[4]:


df_models = get_models_data_frame()
try:
    df_show = df_models.sort_values(by=['wbss', 'bleu'], ascending=False).head()
except KeyError: #if no scode yet
    df_show = df_models
    df_show['wbss'] = np.nan
    df_show['bleu'] = np.nan

    
df_show.tail()


# In[5]:


import logging
import  vqa_logger 
logger = logging.getLogger(__name__)
import IPython


# In[6]:


model_id = 5#int(model_id)
model_folder = 'C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190315_1823_38\\'

question_category = 'Abnormality'
data_access = SpecificDataAccess(data_access_api.folder, question_category=question_category, group=None)
# SpecificDataAccess?


model_arg = model_folder
mp = DefaultVqaModelPredictor(model_arg,data_access=data_access)
mp


# In[7]:


mp.df_validation.head(2)


# In[8]:


code = get_highlighted_function_code(mp.predict,remove_comments=False)
IPython.display.display(code)


# In[9]:


df_validation = mp.df_validation
df_train = data_access.load_processed_data(group='train')


# In[10]:


validation_set = df_train


# In[11]:


validation_set.sample(3)


# In[12]:


df_predictions = mp.predict(validation_set)
df_predictions.sample(5)


# In[ ]:


df_predictions.describe()


# #### Take a look at results for a single image:

# In[ ]:



image_name = df_predictions.image_name.sample(1).values[0]

df_image = df_predictions[df_predictions.image_name == image_name]
# print(f'Result: {set(df_image.prediction)}')

image_path = df_image.path.values[0]

def get_row_evaluation(row, metric):
    return VqaMedEvaluatorBase.get_all_evaluation(predictions=[row.prediction], ground_truth=[row.answer])[metric]

sorted_cols = sorted(df_image.columns, key=lambda s: s not  in ['answer', 'prediction'])
df_image = df_image[sorted_cols]

df_image['wbss'] = df_image.apply(lambda row: get_row_evaluation(row, 'wbss'), axis=1)
df_image['bleu'] = df_image.apply(lambda row: get_row_evaluation(row, 'bleu'), axis=1)
df_image


# In[ ]:


from IPython.display import Image, HTML, display_html
image = Image(filename = image_path, width=600, height=600)
image


# ## Evaluating the Model

# In[ ]:


validation_prediction = df_predictions
predictions = validation_prediction.prediction.values
ground_truth = validation_prediction.answer.values
results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
print(f'Got results of\n{results}')


# In[ ]:


# validation_prediction.sort_values(by=['probabilities'], ascending=False)


# In[36]:


model_db_id = mp.model_idx_in_db
assert model_db_id >= 0 
model_db_id


# In[17]:


bleu = results['bleu']
wbss = results['wbss']
model_score = ModelScore(model_db_id, bleu=bleu, wbss=wbss)
model_score


# ##### Add the score to DB:

# In[18]:


# DAL.insert_dal(model_score)

