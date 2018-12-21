
# coding: utf-8

# In[15]:


import os
import pandas as pd
from pandas import HDFStore
from nltk.corpus import stopwords


# In[16]:


from common.constatns import data_location, vqa_specs_location, fn_meta
from common.settings import embedding_dim, seq_length
from common.classes import VqaSpecs
from common.utils import VerboseTimer


# ### Preprocessing and creating meta_loc data

# Get the data itself, Note the only things required in dataframe are:
# 1. image_name
# 2. question
# 3. answer
# 

# In[17]:


print(f'loading from:\n{data_location}')
with VerboseTimer("Loading Data"):
    with HDFStore(data_location) as store:
         df_data = store['data']
        
df_data = df_data[df_data.group.isin(['train','validation'])]
print(f'Data length: {len(df_data)}')        
df_data.head(2)


# In[18]:


import numpy as np
d = df_data[df_data.imaging_device.isin(['ct','mri'])]
print(np.unique(df_data.imaging_device))
print(np.unique(d.imaging_device))


# #### We will use this function for creating meta_loc data:

# In[19]:


from vqa_logger import logger 
import itertools
import string
from common.os_utils import File #This is a simplehelper file of mine...

def create_meta(df, hdf_output_location):
        
        print(f"Dataframe had {len(df)} rows")
        english_stopwords = set(stopwords.words('english'))
        def get_unique_words(col):           
            single_string = " ".join(df[col])
            exclude = set(string.punctuation)
            s_no_panctuation = ''.join(ch.lower() for ch in single_string if ch not in exclude)
            unique_words = set(s_no_panctuation.split(" ")).difference({'',' '})            
            unique_words = unique_words.difference(english_stopwords)
            print("column {0} had {1} unique words".format(col,len(unique_words)))
            return unique_words

        cols = ['question', 'answer']
        df_unique_words = set(itertools.chain.from_iterable([get_unique_words(col) for col in cols]))
        df_unique_answers = set([ans.lower() for ans in df['answer']])        
        
        df_unique_imaging_devices = set(df['imaging_device'])
        unknown_devices = ['both', 'unknown']
        df_unique_imaging_devices = [v for v in df_unique_imaging_devices if v not in unknown_devices]
        

        words = sorted(list(df_unique_words), key=lambda w: (len(w),w))
        words = [w for w in words if 
                 w in ['ct', 'mri'] 
                 or len(w) >=3 
                 and not w[0].isdigit() ]
        
        metadata_dict = {}       
        metadata_dict['words'] = {'word': words}            
        metadata_dict['answers'] = {'answer':list(df_unique_answers)}            
        metadata_dict['imaging_devices'] = {'imaging_device': df_unique_imaging_devices}
            

        try:
            os.remove(hdf_output_location)
        except OSError:
            pass
        
        for name, dictionary in metadata_dict.items():
            df_curr = pd.DataFrame(dictionary,dtype=str)
            df_curr.to_hdf(hdf_output_location, name, format='table')
            

        
        with HDFStore(hdf_output_location) as metadata_store:           
            print("Meta number of unique answers: {0}".format(len(metadata_store['answers'])))
            print("Meta number of unique words: {0}".format(len(metadata_store['words'])))

#         df_ix_to_word = pd.DataFrame.from_dict(metadata['ix_to_word'])
#         light.to_hdf(data_location, 'light', mode='w', data_columns=['image_name', 'imaging_device', 'path'], format='table')

        return metadata_store
        


# In[20]:


print("----- Creating meta_loc -----")
meta_data = create_meta(df_data, fn_meta)

with HDFStore(fn_meta) as metadata_store:           
    df_words = metadata_store['words']
    df_answers = metadata_store['answers']
    df_imaging_device = metadata_store['imaging_devices']
    
df_words.head()


# #### Saving the data, so later on we don't need to compute it again

# In[21]:


def get_vqa_specs(meta_location):    
    dim = embedding_dim
    s_length = seq_length    
    return VqaSpecs(embedding_dim=dim, 
                    seq_length=s_length, 
                    data_location=os.path.abspath(data_location),
                    meta_data_location=os.path.abspath(meta_location))

vqa_specs = get_vqa_specs(fn_meta)

# Show waht we got...
vqa_specs


# In[22]:


File.dump_pickle(vqa_specs, vqa_specs_location)
logger.debug(f"VQA Specs saved to:\n{vqa_specs_location}")


# ##### Test Loading:

# In[23]:


loaded_vqa_specs = File.load_pickle(vqa_specs_location)
loaded_vqa_specs


# In[24]:


print (f"vqa_specs_location = '{vqa_specs_location}'".replace('\\','\\\\'))

