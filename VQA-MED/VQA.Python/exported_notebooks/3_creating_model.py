
# coding: utf-8

# In[1]:


from classes.vqa_model_builder import VqaModelBuilder
from common.utils import VerboseTimer

import IPython
from IPython.display import Image, display
from common.functions import get_highlighted_function_code


# In[2]:


# TODO: Do we need this?
from keras import backend as keras_backend
keras_backend.clear_session()


# In[3]:


# loss, activation = 'categorical_crossentropy', 'softmax' # good for a model to predict multiple mutually-exclusive classes:
# loss, activation = 'binary_crossentropy', 'sigmoid'
loss, activation = 'categorical_crossentropy', 'sigmoid'

with VerboseTimer("Instantiating VqaModelBuilder"):
    mb = VqaModelBuilder(loss, activation)


# #### What does the data looks like?

# In[4]:


mb.df_meta_answers
mb.df_meta_words
mb.df_meta_imaging_devices
mb.df_meta_answers.tail(2)


# #### Before we start, lets take a look at the functions that will create the model:

# ##### word_2_vec_model
# Define how to build the word-to vector branch:

# In[5]:


code = get_highlighted_function_code(VqaModelBuilder.word_2_vec_model, remove_comments=True)
IPython.display.display(code)  


# ##### get_image_model:
# In the same manner, defines how to build the image representation branch:

# In[6]:


code = get_highlighted_function_code(VqaModelBuilder.get_image_model, remove_comments=False)
IPython.display.display(code)  


# ##### And the actual function for getting the model:

# In[7]:


code = get_highlighted_function_code(mb.get_vqa_model, remove_comments=True)
IPython.display.display(code)  


# ### Creating the model

# In[8]:


with VerboseTimer("Gettingt the model"):
    model = mb.get_vqa_model()


# #### We better save it:

# ##### The saving function:

# In[9]:


code = get_highlighted_function_code(VqaModelBuilder.save_model, remove_comments=False)
IPython.display.display(code)  


# In[10]:


model_fn, summary_fn, fn_image = VqaModelBuilder.save_model(model)


# ### Display a plot + summary:

# #### Where are the trainable parameters?

# ##### The finction:

# In[11]:


code = get_highlighted_function_code(VqaModelBuilder.get_trainable_params_distribution, remove_comments=False)
IPython.display.display(code)


# In[12]:


top = VqaModelBuilder.get_trainable_params_distribution(model)


# In[13]:


display(Image(filename=fn_image))
model.summary()


# Copy these items to the next notebook of training the model

# In[14]:


msg = f"Summary: {summary_fn}\n"
msg += f"Image: {fn_image}\n"
location_message = f"model_location = '{model_fn}'"


print(msg)
print(location_message.replace('\\', '\\\\'))

