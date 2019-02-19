#!/usr/bin/env python
# coding: utf-8

# ### Training the model

# In[5]:


import IPython
from classes.vqa_model_trainer import VqaModelTrainer
from common.model_utils import get_trainable_params_distribution
from common.functions import get_highlighted_function_code
from common.settings import data_access
from common.utils import VerboseTimer
from data_access.model_folder import ModelFolder


# In[6]:


import logging
import vqa_logger 
logger = logging.getLogger(__name__)


# In[7]:


model_location = 'C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190218_1955_22\\'
model_folder = ModelFolder(model_location)
model_folder


# ### Loading the model to train:

# In[ ]:


epochs = 1
batch_size = 256
mt = VqaModelTrainer(model_folder, use_augmentation=True,batch_size=batch_size, data_access=data_access)


# #### Lets take a look at the parameters:

# In[ ]:


get_trainable_params_distribution(mt.model)
# mt.model.summary()


# #### And a look at data:

# In[ ]:


mt.df_meta_answers
mt.df_meta_words
mt.df_meta_answers.tail(2)


# In[ ]:


logger.debug(f"train Shape: {mt.data_train.shape}")
logger.debug(f"validation Shape: {mt.data_val.shape}")
mt.data_train.head(0)


# ### Overview of preperations for training:

# ##### The functions for getting the features & labels:

# In[ ]:


from common.functions import get_features, sentences_to_hot_vector, hot_vector_to_words
code_get_labels = get_highlighted_function_code(mt.get_labels, remove_comments=True)


code_get_features = get_highlighted_function_code(get_features, remove_comments=True)
code_hot_vector = get_highlighted_function_code(sentences_to_hot_vector, remove_comments=True)


print('Getting the label using a hot vector\n')
IPython.display.display(code_get_labels)
print('\n\nThe underlying method:\n')
IPython.display.display(code_hot_vector)


print('\n\nGetting the features using question embeding concatenation:\n')
IPython.display.display(code_get_features)


# ##### Example of hot vector of answer (AKA answer...)

# In[ ]:


df = mt.data_train

class_df = mt.class_df
class_count = len(class_df)
# class_df.sample(5)

classes_indices_df = [class_df.loc[class_df.isin(ans.lower().split())] for ans in  df.answer]
classes_indices = [list(d.index) for d in classes_indices_df]

idx_sample = 9
print(df.answer[idx_sample])
classes_indices[idx_sample]


# ##### Will transform the sentences into vector and back using the following:

# In[ ]:


code = get_highlighted_function_code(hot_vector_to_words,remove_comments=False)
IPython.display.display(code)  


# ##### Check it looks sane by inversing the binarizing:

# In[ ]:


# words = mt.df_meta_words.word

arr_one_hot_vector = mt.get_labels(mt.data_train)
categorial_labels = arr_one_hot_vector

idx = 0
answer =  mt.data_train.answer.loc[idx]
print(f'The sentence:\n{answer}')

one_hot_vector = arr_one_hot_vector[idx]
label_words = hot_vector_to_words(one_hot_vector, mt.class_df)
print('\n\nThe reversed answer from hot vector:')
label_words


# In[ ]:


history = mt.train()


# ### Save trained model:

# In[ ]:


with VerboseTimer("Saving trained Model"):
    model_folder = mt.save(mt.model, mt.model_folder, history)


# In[ ]:


print (model_folder.model_path)

