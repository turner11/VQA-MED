
# coding: utf-8

# In[1]:


from common.DAL import get_models_data_frame, get_model
df_models = get_models_data_frame()
df_models.head()


# In[2]:


model_id = max(df_models.id)#2

notes = df_models.loc[df_models.id == model_id].notes.values[0]

print(f'Getting model #{model_id} ({notes})')
model_dal = get_model(model_id)
model_dal


# In[3]:


#From Step #2:
vqa_specs_location = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\vqa_specs.pkl'                     


# In[4]:


model_location = model_dal.model_location
model_location
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180629_1220_23\\vqa_model_ClassifyStrategies.NLP_trained.h5'


# In[5]:


# %%capture
import os
import numpy as np
from pandas import HDFStore
from vqa_logger import logger 
from enum import Enum
import IPython
from keras.models import load_model


# In[6]:


from common.constatns import images_path_test
from common.utils import VerboseTimer
from parsers.VQA18 import Vqa18Base
from common.functions import get_size, get_highlited_function_code, normalize_data_strucrture
from common.functions import get_highlited_function_code, get_features, _concat_row, predict
from vqa_logger import logger
from common.os_utils import File 


# In[7]:


with VerboseTimer("Loading Model"):
    model = load_model(model_location)


# In[8]:


vqa_specs = File.load_pickle(vqa_specs_location)
data_location = vqa_specs.data_location
data_location


# In[14]:


meta_data = vqa_specs.meta_data


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


# In[15]:


df_predictions = predict(model, df_data, meta_data)
df_predictions.head()


# In[16]:


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


# In[ ]:


validation_prediction = predict(model, df_validation, meta_data)


# In[25]:


validation_prediction.head(2)


# ## Evaluating the Model

# In[28]:


from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
predictions = validation_prediction.prediction.values
ground_truth = validation_prediction.answer
results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
results

