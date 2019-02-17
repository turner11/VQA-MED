#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


from classes.vqa_model_predictor import VqaModelPredictor, DefaultVqaModelPredictor
from common.DAL import get_models_data_frame, get_model
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.functions import get_highlighted_function_code
import vqa_logger 


# In[8]:


df_models = get_models_data_frame()
try:
    df_show = df_models.sort_values(by=['wbss', 'bleu'], ascending=False).head()
except KeyError: #if no scode yet
    df_show = df_models
    df_show['wbss'] = np.nan
    df_show['bleu'] = np.nan

    
df_show.tail()


# In[9]:


import logging
import  vqa_logger 
logger = logging.getLogger(__name__)
import IPython


# In[10]:


known_good_model = 163#85
model_id = known_good_model #df_show.id.iloc[0]
model_id = 1#int(model_id)
mp = DefaultVqaModelPredictor(model_id)
mp


# In[ ]:


mp.df_validation.head(2)


# In[ ]:


code = get_highlighted_function_code(mp.predict,remove_comments=False)
IPython.display.display(code)


# In[ ]:


df_data = mp.df_validation
df_predictions = mp.predict(mp.df_validation)
df_predictions.head()


# In[ ]:


df_predictions.describe()


# In[ ]:


idx = 42
image_names = df_predictions.image_name.values
image_name = image_names[idx]

df_image = df_predictions[df_predictions.image_name == image_name]
# print(f'Result: {set(df_image.prediction)}')

image_path = df_image.path.values[0]
df_image


# In[ ]:


from IPython.display import Image
Image(filename = image_path, width=400, height=400)


# In[ ]:


df_image = df_data[df_data.image_name == image_name].copy().reset_index()
image_prediction = mp.predict(df_image)
image_prediction


# ## Evaluating the Model

# In[ ]:


validation_prediction = df_predictions
predictions = validation_prediction.prediction.values
ground_truth = validation_prediction.answer.values
results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
print(f'Got results of\n{results}')


# In[ ]:


validation_prediction.head(2)

