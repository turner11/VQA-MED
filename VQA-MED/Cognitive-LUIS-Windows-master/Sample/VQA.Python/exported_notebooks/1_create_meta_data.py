
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[2]:


# Pre process results files
fn_meta            = os.path.abspath('data/meta_data.json')


# In[3]:


from common.constatns import train_data, validation_data


# ### Preprocessing and creating meta data

# Get the data itself, Note the only things required in dataframe are:
# 1. image_name
# 2. question
# 3. answer
# 

# In[4]:


from parsers.VQA18 import Vqa18Base
df_train = Vqa18Base.get_instance(train_data.processed_xls).data            
df_val = Vqa18Base.get_instance(validation_data.processed_xls).data


# We will use this function for creating meta data:

# In[5]:


from vqa_logger import logger 
import itertools
import string
from common.os_utils import File #This is a simplehelper file of mine...

def create_meta(meta_file_location, df):
        logger.debug("Creating meta data ('{0}')".format(meta_file_location))
        print(f"Dataframe had {len(df)} rows")
        def get_unique_words(col):
            single_string = " ".join(df[col])
            exclude = set(string.punctuation)
            s_no_panctuation = ''.join(ch for ch in single_string if ch not in exclude)
            unique_words = set(s_no_panctuation.split(" ")).difference({'',' '})
            print("column {0} had {1} unique words".format(col,len(unique_words)))
            return unique_words

        cols = ['question', 'answer']
        df_unique_words = set(itertools.chain.from_iterable([get_unique_words(col) for col in cols]))
        df_unique_answers = set(df['answer'])        

        metadata = {}
        metadata['ix_to_word'] = {str(word): int(i) for i, word in enumerate(df_unique_words)}
        metadata['ix_to_ans'] = {i:ans for i, ans in enumerate(df_unique_answers)}
        metadata['ans_to_ix'] = {ans:i for i, ans in enumerate(df_unique_answers)}
                
        
        #------------------- Asserts
        answers = metadata['ix_to_ans'].values()
        words = metadata['ix_to_word'].values()
        
        assert len(set(answers)) == len(answers), 'Got duplicate answers'
        assert len(set(words)) == len(words), 'Got duplicate words'        
        
        print("Meta number of unique answers: {0}".format(len(set(metadata['ix_to_ans'].values()))))
        print("Meta number of unique words: {0}".format(len(set(metadata['ix_to_word'].values()))))

        File.dump_json(metadata,meta_file_location)
        return metadata


# In[6]:


print("----- Creating meta -----")
full_df = pd.concat([df_train, df_val])
meta_data = create_meta(fn_meta, full_df)
print(f"Meta file available at: {fn_meta}")
# meta_data

