#!/usr/bin/env python
# coding: utf-8

# # Creating Meta data

# In this note book we will create the meta data.  
# Meta data holds information about which unique words and answers exists in training & validation datasets, and in which categories they appeared.  
# This information will allow us later on to build dedicated models for each category

# ### Some main functions we used:

# In[1]:


import IPython
from common.functions import get_highlighted_function_code


# In[2]:


from pre_processing.meta_data import create_meta
code = get_highlighted_function_code(create_meta,remove_comments=False)
IPython.display.display(code)  


# ---
# ## The code:

# In[3]:


import os
import pandas as pd
pd.set_option('display.max_colwidth', -1)


# In[4]:


from common.settings import data_access
import vqa_logger 
from pre_processing.meta_data import create_meta


# Creating the meta data. Note the only things required in the input dataframe are:
# 1. image_name
# 2. processed question
# 3. processed answer
# 

# In[6]:


# index	image_name	question	answer	group	path	original_question	original_answer	tumor	hematoma	brain	abdomen	neck	liver	imaging_device	answer_embedding	question_embedding	is_imaging_device_question
df_data = data_access.load_processed_data(columns=['path','question','answer', 'processed_question','processed_answer', 'group','question_category'])        
df_data = df_data[df_data.group.isin(['train','validation', 'test'])]
print(f'Data length: {len(df_data)}')        


# The input data:

# In[7]:


df_data.sample(2)


# In[8]:


print("----- Creating meta -----")
meta_data_dict = create_meta(df_data)


# #### Saving the data, so later on we don't need to compute it again

# In[7]:


print("----- Saving meta -----")
data_access.save_meta(meta_data_dict)


# ##### Test Loading:

# In[9]:


loaded_meta = data_access.load_meta()
answers_meta = loaded_meta['answers']
words_meta = loaded_meta['words']


answers_meta.question_category.describe()
# words_meta.question_category.describe()

# answers_meta.sample(5)
# words_meta.sample(5)

# words_meta.question_category.drop_duplicates()


# View the data:

# In[63]:


from IPython.display import display_html
def display_side_by_side(*data_frames):
    html_str=''
    for df in data_frames:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

display_side_by_side(answers_meta.sample(5), words_meta.sample(5))   

