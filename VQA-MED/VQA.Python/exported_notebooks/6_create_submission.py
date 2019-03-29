#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import pandas as pd
from classes.vqa_model_predictor import DefaultVqaModelPredictor
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.utils import VerboseTimer
import vqa_logger 
import logging
logger = logging.getLogger(__name__)


# In[24]:


mp = DefaultVqaModelPredictor.get_contender()

df_to_predict = mp.df_test
# df_to_predict = mp.df_validation
with VerboseTimer(f"Predictions for VQA contender"):
    df_predictions = mp.predict(df_to_predict)


predictions = df_predictions.prediction.values
predictions[:5]


# In[25]:


df_output = df_to_predict.copy()
df_output['image_id'] = df_output.path.apply(lambda p: p.rsplit(os.sep)[-1].rsplit('.', 1)[0])
df_output['prediction'] = predictions

columns_to_remove = ['path',  'answer_embedding', 'question_embedding', 'group', 'diagnosis', 'processed_answer']
for col in columns_to_remove:
    del df_output[col]

sort_columns = sorted(df_output.columns, key=lambda c: c not in ['question', 'prediction', 'answer'])
df_output = df_output[sort_columns]    
df_output.sample(10)


# In[26]:


len(df_output), len(df_output.image_id.drop_duplicates())


# In[7]:


strs = []

for i, (idx, row) in enumerate(df_output.iterrows()):
    image = row["path"].rsplit('\\')[-1].rsplit('.', 1)[0]
    #s = f'{i + 1}\t{image}\t{predictions[i]}'
    s = f'{i + 1}\t{image}\t{predictions[i]}'
    strs.append(s)

res = '\n'.join(strs)


# In[8]:


mp


# In[10]:


print(res)

