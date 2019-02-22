#!/usr/bin/env python
# coding: utf-8

# In[1]:


from classes.vqa_model_builder import VqaModelBuilder
from common.utils import VerboseTimer

import IPython
from IPython.display import Image, display
from common.functions import get_highlighted_function_code
import vqa_logger 


# In[2]:


# TODO: Do we need this?
from keras import backend as keras_backend
keras_backend.clear_session()


# In[3]:


# loss, activation = 'categorical_crossentropy', 'softmax' # good for a model to predict multiple mutually-exclusive classes:
# loss, activation = 'binary_crossentropy', 'sigmoid'
# loss, activation = 'categorical_crossentropy', 'sigmoid'
# loss, activation = 'cosine_proximity', 'relu'
loss, activation, lstm_units = 'cosine_proximity', 'tanh', 0


categorical_data_frame_name = 'answers'
categorical_data_frame_name = 'words'

with VerboseTimer("Instantiating VqaModelBuilder"):
    mb = VqaModelBuilder(loss, activation, lstm_units=lstm_units, categorical_data_frame_name=categorical_data_frame_name)


# #### What does the data looks like?

# In[4]:


mb.categorical_data_frame.sample(5)


# #### Before we start, lets take a look at the functions that will create the model:

# ##### word_2_vec_model
# Define how to build the word-to vector branch:

# In[5]:


code = get_highlighted_function_code(VqaModelBuilder.word_2_vec_model,remove_comments=True)
IPython.display.display(code)  


# ##### get_image_model:
# In the same manner, defines how to build the image representation branch:

# In[6]:


code = get_highlighted_function_code(VqaModelBuilder.get_image_model,remove_comments=False)
IPython.display.display(code)  


# ##### And the actual function for getting the model:

# In[7]:


code = get_highlighted_function_code(mb.get_vqa_model,remove_comments=True)
IPython.display.display(code)  


# ### Creating the model

# In[8]:


with VerboseTimer("Gettingt the model"):
    model = mb.get_vqa_model()


# #### We better save it:

# ##### The saving function:

# In[9]:


code = get_highlighted_function_code(VqaModelBuilder.save_model,remove_comments=False)
IPython.display.display(code)  


# In[10]:


model_folder = VqaModelBuilder.save_model(model, mb.categorical_data_frame_name)


# ### Display a plot + summary:

# #### Where are the trainable parameters?

# ##### The function:

# In[11]:


code = get_highlighted_function_code(VqaModelBuilder.get_trainable_params_distribution,remove_comments=False)
IPython.display.display(code)


# In[12]:


top = VqaModelBuilder.get_trainable_params_distribution(model)


# In[13]:


display(Image(filename=str(model_folder.image_file_path)))
model.summary()


# Copy these items to the next notebook of training the model

# In[14]:


msg = f"Summary: {str(model_folder.summary_path)}\n"
msg += f"Image: {model_folder.image_file_path}\n"
location_message = f"model_location = '{model_folder.model_path}'"


print(msg)
print(location_message.replace('\\', '\\\\'))

