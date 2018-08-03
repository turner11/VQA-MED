
# coding: utf-8

# In[1]:


from common.DAL import get_models_data_frame, get_model
df_models = get_models_data_frame()
df_models.head()


# In[2]:


model_id = 1
model_dal = get_model(model_id)
model_dal


# In[3]:


#From Step #2:
vqa_specs_location = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\data\\vqa_specs.pkl'


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
from vqa_logger import logger
from common.os_utils import File 


# In[7]:


with VerboseTimer("Loading Model"):
    model = load_model(model_location)


# In[8]:


vqa_specs = File.load_pickle(vqa_specs_location)
data_location = vqa_specs.data_location
data_location


# In[9]:


code = get_highlited_function_code(normalize_data_strucrture,remove_comments=True)
IPython.display.display(code)


# In[10]:


logger.debug(f"Loading test data from {data_location}")
with VerboseTimer("Loading Test Data"):
    with HDFStore(data_location) as store:        
        df_data = store['test']


# In[16]:


df_data.head(2)


# ## TODO: Duplicate:

# In[12]:


def concate_row(df, col):
    return np.concatenate(df[col], axis=0)

def get_features_and_labels(df):
    image_features = np.asarray([np.array(im) for im in df['image']])
    # np.concatenate(image_features['question_embedding'], axis=0).shape
    question_features = concate_row(df, 'question_embedding') 

    reshaped_q = np.array([a.reshape(a.shape + (1,)) for a in question_features])
    
    features = ([f for f in [reshaped_q, image_features]])    
    
    return features
    
    


# In[13]:


features = get_features_and_labels(df_data)


# In[14]:



p = model.predict(features)


# In[15]:


p


# In[21]:


predictions = [np.argmax(a, axis=None, out=None) for a in p]
predictions[:10]


# In[29]:


meta_data = vqa_specs.meta_data
ix_to_img_device = meta_data['ix_to_img_device']
results = [ix_to_img_device[i] for i in predictions]
results[:10]

list(zip(df_data.image_name.values, results))[:10]


# In[79]:


# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
idx = 42
image_names = df_data.image_name.values
image_name = image_names[idx]

print(f'Result: {results[idx]}')
idxs = [index for index, value in enumerate(image_names) if value == image_name]
all_results_for_image = {results[idx] for idx in idxs}
print(f'All results for image: {results[idx]}')
print('DataFrame:')
      
df_image = df_data[df_data.image_name==image_name]

len(image_path)
image_path = df_image['path'].values[0]


df_mini = df_image[['question','answer']]
df_mini


# In[80]:


from IPython.display import Image
Image(filename = image_path, width=400, height=400)

