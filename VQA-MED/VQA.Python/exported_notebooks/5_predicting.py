
# coding: utf-8

# In[1]:


from common.DAL import get_models_data_frame, get_model_by_id
df_models = get_models_data_frame()
df_models


# In[2]:


model_id = max(df_models.id)#2

notes = df_models.loc[df_models.id == model_id].notes.values[0]

print(f'Getting model #{model_id} ({notes})')
model_dal = get_model_by_id(model_id)
model_dal


# In[3]:


model_location = model_dal.model_location
model_location
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180629_1220_23\\vqa_model_ClassifyStrategies.NLP_trained.h5'


# In[4]:


# %%capture
import os
import numpy as np
from pandas import HDFStore
from vqa_logger import logger 
from enum import Enum
import IPython
from keras.models import load_model


# In[5]:


from common.constatns import images_path_test, vqa_specs_location
from common.utils import VerboseTimer
from parsers.VQA18 import Vqa18Base
from common.functions import get_size, get_highlited_function_code, normalize_data_strucrture
from common.functions import get_highlited_function_code, get_features, concat_row, predict
from vqa_logger import logger
from common.os_utils import File

from evaluate.statistical import f1_score, recall_score, precision_score


# In[6]:


with VerboseTimer("Loading Model"):    
    model = load_model(model_location, custom_objects= {'f1_score': f1_score, 'recall_score':recall_score, 'precision_score':precision_score})
#     model = load_model(model_location)


# In[7]:


vqa_specs = File.load_pickle(vqa_specs_location)
data_location = vqa_specs.data_location
data_location


# In[8]:


meta_data_location = vqa_specs.meta_data_location


# In[9]:


code = get_highlited_function_code(normalize_data_strucrture,remove_comments=True)
IPython.display.display(code)


# In[10]:


logger.debug(f"Loading test data from {data_location}")
with VerboseTimer("Loading Test Data"):
    with HDFStore(data_location) as store:        
        df_data = store['test']
        df_training = store['data']
# The validation is for evaluating
df_validation = df_training[df_training.group == 'validation'].copy()
del df_training


# In[11]:


df_data.head(2)


# In[12]:


import importlib
import common
importlib.reload(common.functions)


code = get_highlited_function_code(predict,remove_comments=False)
IPython.display.display(code)


# In[13]:


# df_data = df_data[df_data.index < 5]
# df_data[['image_name','question','answer','group']].head(1)


# In[14]:


# import pandas as pd
# # model, df_data, meta_data_location
# with VerboseTimer("Just a place holder"):
# # def apredict(model, df_data: pd.DataFrame, meta_data_location=None):
#     PERCENTILE =99.8
#     # predict
#     features = get_features(df_data)        
#     p = model.predict(features)
    
   
#     percentiles = [np.percentile(curr_pred, PERCENTILE) for curr_pred in p]
#     #pass_vals = [ (i,[curr_pred for curr_pred  in curr_pred_arr if curr_pred >= curr_percentile]) for (i, curr_pred_arr), curr_percentile in zip(enumerate(p), percentiles)]
#     enumrated_p = [[(i,v) for i,v in enumerate(curr_p)] for curr_p in p]
#     pass_vals = [ ([(i, curr_pred) for i , curr_pred  in curr_pred_arr if curr_pred >= curr_percentile]) for curr_pred_arr, curr_percentile in zip(enumrated_p, percentiles)]

#     #[(i,len(curr_pass_arr)) for i, curr_pass_arr in  pass_vals]

#     # vector-to-value
# #     predictions = [np.argmax(a, axis=None, out=None) for a in p]
#     predictions = [i for curr_pass_arr in  pass_vals for i, curr_p in curr_pass_arr]
#     results = [curr_p for curr_pass_arr in  pass_vals for i, curr_p in curr_pass_arr]

#     # dictionary for creating a data frame
#     cols_to_transfer = ['image_name', 'question', 'answer', 'path']
#     df_dict = {col_name: df_data[col_name] for col_name in cols_to_transfer}

#     if meta_data_location:
#         df_meta_words = pd.read_hdf(meta_data_location, 'words')
#         results = df_meta_words.loc[predictions]

#         imaging_device_probabilities = {row.word: [prediction[index] for prediction in p] for index, row in results.iterrows()}
#         df_dict.update(imaging_device_probabilities)

#     df_dict['prediction'] =  " ".join([r for r in results.word.values])
#     df = pd.DataFrame(df_dict)

#     # Arranging in a prettier way
#     sort_columns = ['image_name', 'question', 'answer', 'prediction']
#     oredered_columns = sorted(df.columns, key=lambda v: v in sort_columns, reverse=True)
#     df = df[oredered_columns]
# df


# In[15]:


# idx = 3
# e = sorted(enumerate(p[idx]), key=lambda tpl: tpl[1], reverse=True)
# e[:5]


# In[16]:


# PERCENTILE = 99.8
# PERCENTILE = 95
# percentiles = [np.percentile(curr_pred, PERCENTILE) for curr_pred in p]
# pass_vals = [ (i,[curr_pred for curr_pred  in curr_pred_arr if curr_pred >= curr_percentile]) for i, (curr_pred_arr, curr_percentile) in enumerate(zip(p, percentiles))]
# [(i,len(curr_pass_arr)) for i, curr_pass_arr in  pass_vals]


# In[17]:


# idx = 3
# min(p[idx]),max(p[idx])


# In[18]:


df_predictions = predict(model, df_data, meta_data_location)
df_predictions.head()


# In[19]:


df_predictions.describe()


# In[20]:


# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
idx = 42
image_names = df_data.image_name.values
image_name = image_names[idx]

df_image = df_predictions[df_predictions.image_name == image_name]
print(f'Result: {set(df_image.prediction)}')

image_path = df_image.path.values[0]
df_image


# In[21]:


from IPython.display import Image
Image(filename = image_path, width=400, height=400)


# In[22]:


validation_prediction = predict(model, df_validation, meta_data_location)


# In[23]:


validation_prediction.head(2)


# ## Evaluating the Model

# In[24]:


from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
predictions = validation_prediction.prediction.values
ground_truth = validation_prediction.answer
results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
results

