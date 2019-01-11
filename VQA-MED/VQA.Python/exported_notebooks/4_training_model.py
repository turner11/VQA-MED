
# coding: utf-8

# ### Training the model

# In[1]:


import IPython
from classes.vqa_model_trainer import VqaModelTrainer
from common.model_utils import get_trainable_params_distribution
import logging
logger = logging.getLogger(__name__)
from common.functions import get_highlited_function_code


# In[2]:


## VGG all words are Classes (Trainable params: 1,070,916). 'categorical_crossentropy', 'sigmoid' .With f1_score, recall_score, precision_score + accuracy metrics
model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180831_1244_55\\vqa_model_.h5'
## VGG all words are Classes (Trainable params: 1,070,916). 'categorical_crossentropy', 'softmax' .With f1_score, recall_score, precision_score + accuracy metrics
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180829_0830_48\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG all words are Classes (Trainable params: 1,070,916) With f1_score, recall_score, precision_score + accuracy metrics
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180828_2149_37\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG all words are Classes (Trainable params: 1,070,916)
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180827_1502_41\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG 2 Classes (Trainable params: 165,762)
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180814_2035_20\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

## VGG 4 Classes
# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180730_0648_46\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

# model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180728_2248_02\\vqa_model_CATEGORIAL.h5'
# strategy_str = 'CATEGORIAL'

# ## Resnet 50: 
# trained_model_location = 'C:\Users\Public\Documents\Data\2018\vqa_models\20180730_0524_48\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5'
# loss: 0.1248 - acc: 0.9570 - val_loss: 2.7968 - val_acc: 0.5420
# Training Model: 12:22:54.619203                


# ### Loading the model to train:

# In[3]:


epochs = 1
batch_size = 75
mt = VqaModelTrainer(model_location, epochs=epochs, batch_size=batch_size)


# #### Lets take a look at the parameters:

# In[4]:


get_trainable_params_distribution(mt.model)
# mt.model.summary()


# #### And a look at data:

# In[5]:


mt.df_meta_answers
mt.df_meta_words
mt.df_meta_imaging_devices
mt.df_meta_answers.tail(2)


# In[6]:


logger.debug(f"train Shape: {mt.data_train.shape}")
logger.debug(f"validation Shape: {mt.data_val.shape}")
mt.data_train.head(0)


# ### Overview of preperations for training:

# ##### The functions for getting the features & labels:

# In[7]:


from common.functions import get_features, _concat_row, sentences_to_hot_vector, hot_vector_to_words
code_get_labels = get_highlited_function_code(mt.get_labels, remove_comments=True)


code_get_features = get_highlited_function_code(get_features, remove_comments=True)
code_concat = get_highlited_function_code(_concat_row, remove_comments=True)
code_hot_vector = get_highlited_function_code(sentences_to_hot_vector, remove_comments=True)


print('Getting the label using a hot vector\n')
IPython.display.display(code_get_labels)
print('\n\nThe underlying method:\n')
IPython.display.display(code_hot_vector)


print('\n\nGetting the features using question embeding concatenation:\n')
IPython.display.display(code_get_features)
IPython.display.display(code_concat)



# ##### Example of hot vector of anser (AKA answer...)

# In[8]:


df = mt.data_train

class_df = mt.class_df
class_count = len(class_df)
# class_df

classes_indices_df = [class_df.loc[class_df.word.isin(ans.lower().split())] for ans in  df.answer]
classes_indices = [list(d.index) for d in classes_indices_df]

idx_sample = 9
print(df.answer[idx_sample])
classes_indices[idx_sample]


# ##### Will transform the sentences into vector and back using the following:

# In[9]:


code = get_highlited_function_code(hot_vector_to_words,remove_comments=False)
IPython.display.display(code)  


# ##### Check it looks sane by inversing the binarizing:

# In[10]:


words = mt.df_meta_words.word

arr_one_hot_vector = mt.get_labels(mt.data_train)
categorial_labels = arr_one_hot_vector

idx = 0
answer =  mt.data_train.answer.loc[idx]
print(f'The sentence:\n{answer}')

one_hot_vector = arr_one_hot_vector[idx]
label_words = hot_vector_to_words(one_hot_vector, words)
print('\n\nThe highlighed labels:')
label_words


# In[11]:


# from utils.gpu_utils import test_gpu
# test_gpu()


# In[ ]:


mt.train()


# ### Save trained model:

# In[ ]:


with VerboseTimer("Saving trained Model"):
    model_fn, summary_fn, fn_image, fn_history = mt.save()


# In[ ]:


print (model_fn)

