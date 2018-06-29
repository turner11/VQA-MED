
# coding: utf-8

# In[1]:


# %%capture
import os
import numpy as np
from pandas import HDFStore
import spacy
from keras.utils import to_categorical
import cv2

from vqa_logger import logger
from common.os_utils import File



# In[2]:


from common.constatns import train_data, validation_data, data_location, fn_meta, vqa_specs_location
from common.settings import nlp_vector, input_length, embedding_dim, image_size, seq_length
from common.classes import VqaSpecs


# In[3]:


meta_data = File.load_json(fn_meta)


# ### Preparing the data for training

# In[4]:


logger.debug(f'using embedding vector: {nlp_vector }')
nlp = spacy.load('en', vectors=nlp_vector)

# logger.debug(f'vector "{nlp_vector}" loaded')
# logger.debug(f'nlp creating pipe')
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
# logger.debug(f'nlp getting embedding')
# word_embeddings = nlp.vocab.vectors.data
logger.debug(f'Got embedding')


# In[5]:


from parsers.VQA18 import Vqa18Base
df_train = Vqa18Base.get_instance(train_data.processed_xls).data            
df_val = Vqa18Base.get_instance(validation_data.processed_xls).data


# In[6]:


logger.debug('Building input dataframe')
cols = ['image_name', 'question', 'answer']

image_name_question = df_train[cols].copy()
image_name_question_val = df_val[cols].copy()


# ##### This is just for performance and quick debug cycles! remove before actual trainining:

# In[7]:


# image_name_question = image_name_question.head(5)
# image_name_question_val = image_name_question_val.head(5)


# In[8]:



def get_text_features(txt):
    ''' For a given txt, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    tokens = nlp(txt)    
    text_features = np.zeros((1, input_length, embedding_dim))
    
    num_tokens_to_take = min([input_length, len(tokens)])
    trimmed_tokens = tokens[:num_tokens_to_take]
    
    for j, token in enumerate(trimmed_tokens):
        # print(len(token.vector))
        text_features[0,j,:] = token.vector
    # Bringing to shape of (1, input_length * embedding_dim)
    ## ATTN - nlp vector:
    text_features = np.reshape(text_features, (1, input_length * embedding_dim))
    return text_features


def get_image(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''    
    im = cv2.resize(cv2.imread(image_file_name), image_size)

    # convert the image to RGBA
#     im = im.transpose((2, 0, 1))
    return im


def get_categorial_labels(df, meta):
    ans_to_ix = meta['ans_to_ix']
    all_classes =  ans_to_ix.keys()
    data_classes = df['answer']
    class_count = len(all_classes)

    classes_indices = [ans_to_ix[ans] for ans in data_classes]
    categorial_labels = to_categorical(classes_indices, num_classes=class_count)
    
    for i in range(len(categorial_labels)):
        assert np.argmax(categorial_labels[i])== classes_indices[i], 'Expected to get argmax at index of label'
    


    return categorial_labels

categorial_labels_train = get_categorial_labels(df_train, meta_data)
categorial_labels_val = get_categorial_labels(df_val, meta_data)
# categorial_labels_train.shape, categorial_labels_val.shape
del df_train
del df_val


# In[9]:


def pre_process_raw_data(df, images_path):
    df['image_name'] = df['image_name'].apply(lambda q: q if q.lower().endswith('.jpg') else q+'.jpg')

    df['path'] =  df['image_name'].apply(lambda name:os.path.join(images_path, name))

    existing_files = [os.path.join(images_path, fn) for fn in os.listdir(images_path)]
    df = df.loc[df['path'].isin(existing_files)]


    logger.debug('Getting questions embedding')
    df['question_embedding'] = df['question'].apply(lambda q: get_text_features(q))


    logger.debug('Getting answers embedding')
    df['answer_embedding'] = df['answer'].apply(lambda q: get_text_features(q))

    logger.debug('Getting image features')
    df['image'] = df['path'].apply(lambda im_path: get_image(im_path))

    logger.debug('Done')
    return df


# ### Do the actual pre processing
# Note:  
# This might take a while...

# In[10]:


logger.debug('----===== Preproceccing train data =====----')
image_locations = train_data.images_path
image_name_question = pre_process_raw_data(image_name_question, image_locations)


# In[11]:


logger.debug('----===== Preproceccing validation data =====----')
image_locations = validation_data.images_path
image_name_question_val = pre_process_raw_data(image_name_question_val, image_locations)


# In[12]:


image_name_question.head(2)


# #### Saving the data, so later on we don't need to compute it again

# In[13]:


def get_vqa_specs(meta_data):    
    dim = embedding_dim
    s_length = seq_length    
    return VqaSpecs(embedding_dim=dim, seq_length=s_length, data_location=data_location,meta_data=meta_data)

vqa_specs = get_vqa_specs(meta_data)

# Show waht we got...
s = str(vqa_specs)
s[:s.index('meta_data=')+10]


# In[14]:



logger.debug("Save the data")

item_to_save = image_name_question
# item_to_save = image_name_question.head(10)

# remove if exists
try:
    os.remove(data_location)
except OSError:
    pass

with HDFStore(data_location) as store:
    store['train']  = image_name_question
    store['val']  = image_name_question_val
    
item_to_save.to_hdf(vqa_specs.data_location, key='df')    
# store = HDFStore('model_input.h5')
logger.debug(f"Saved to {vqa_specs.data_location}")


# In[15]:


File.dump_pickle(vqa_specs, vqa_specs_location)
logger.debug(f"VQA Specs saved to:\n{vqa_specs_location}")

