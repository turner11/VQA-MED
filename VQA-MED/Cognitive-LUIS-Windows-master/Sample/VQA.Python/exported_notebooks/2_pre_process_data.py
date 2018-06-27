
# coding: utf-8

# In[1]:


import os
import numpy as np
from enum import Enum
import pandas as pd
from pandas import HDFStore
from vqa_logger import logger
from utils.os_utils import File

import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)


# In[2]:


fn_meta            = os.path.abspath('./data/meta_data.json')
data_location      = os.path.abspath('./data/model_input.h5')
vqa_specs_location = os.path.abspath('./data/vqa_specs.pkl')


# In[3]:


meta_data = File.load_json(fn_meta)


# In[4]:


class ClassifyStrategies(Enum):
    NLP = 1
    CATEGORIAL = 2


# ### Preparing the data for training

# In[5]:


# TODO: Duplicate:
spacy_emmbeding_dim = 384

embedding_dim = 384

# input_dim : the vocabulary size. This is how many unique words are represented in your corpus.
# output_dim : the desired dimension of the word vector. For example, if output_dim = 100, then every word will be mapped onto a vector with 100 elements, whereas if output_dim = 300, then every word will be mapped onto a vector with 300 elements.
# input_length : the length of your sequences. For example, if your data consists of sentences, then this variable represents how many words there are in a sentence. As disparate sentences typically contain different number of words, it is usually required to pad your sequences such that all sentences are of equal length. The keras.preprocessing.pad_sequence method can be used for this (https://keras.io/preprocessing/sequence/).
input_length = 32 # longest question / answer was 28 words. Rounding up to a nice round number

# ATTN - nlp vector: Arbitrary  selected for both question and asnwers
embedded_sentence_length = input_length * embedding_dim 


# In[6]:


import os
import spacy

seq_length =    26


vectors = ['en_core_web_lg','en_core_web_md', 'en_core_web_sm']#'en_vectors_web_lg'
vector = vectors[2]
logger.debug(f'using embedding vector: {vector}')
nlp = spacy.load('en', vectors=vector)
# logger.debug(f'vector "{vector}" loaded')
# logger.debug(f'nlp creating pipe')
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
# logger.debug(f'nlpgetting embedding')
# word_embeddings = nlp.vocab.vectors.data
logger.debug(f'Got embedding')




# embedding_dim = 300
# glove_path =                    os.path.abspath('data/glove.6B.{0}d.txt'.format(embedding_dim))
# embedding_matrix_filename =     os.path.abspath('data/ckpts/embeddings_{0}.h5'.format(embedding_dim))
ckpt_model_weights_filename =   os.path.abspath('data/ckpts/model_weights.h5')

spacy_emmbeding_dim = 384

embedding_dim = 384

DEFAULT_IMAGE_WIEGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}

# classify_strategy = ClassifyStrategies.CATEGORIAL
classify_strategy = ClassifyStrategies.NLP


# In[7]:


from collections import namedtuple
dbg_file_csv_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA.csv'
dbg_file_xls_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA_post_pre_process_intermediate.xlsx'#"'C:\\\\Users\\\\avitu\\\\Documents\\\\GitHub\\\\VQA-MED\\\\VQA-MED\\\\Cognitive-LUIS-Windows-master\\\\Sample\\\\VQA.Python\\\\dumped_data\\\\vqa_data.xlsx'
dbg_file_xls_processed_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA_post_pre_process.xlsx'
train_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images\\embbeded_images.hdf'
images_path_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images'


dbg_file_csv_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA.csv'
dbg_file_xls_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA_post_pre_process_intermediate.xlsx'
dbg_file_xls_processed_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA_post_pre_process.xlsx'
validation_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-images\\embbeded_images.hdf'
images_path_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-images'


dbg_file_csv_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA.csv'
dbg_file_xls_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA_post_pre_process_intermediate.xlsx'
dbg_file_xls_processed_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA_post_pre_process.xlsx'
test_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-images\\embbeded_images.hdf'
images_path_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-images'

DataLocations = namedtuple('DataLocations', ['data_tag', 'raw_csv', 'raw_xls', 'processed_xls','images_path'])
train_data = DataLocations('train', dbg_file_csv_train,dbg_file_xls_train,dbg_file_xls_processed_train, images_path_train)
validation_data = DataLocations('validation', dbg_file_csv_validation, dbg_file_xls_validation, dbg_file_xls_processed_validation, images_path_validation)
test_data = DataLocations('test', dbg_file_csv_test, dbg_file_xls_test, dbg_file_xls_processed_test, images_path_test)


# In[8]:


from parsers.VQA18 import Vqa18Base
df_train = Vqa18Base.get_instance(train_data.processed_xls).data            
df_val = Vqa18Base.get_instance(validation_data.processed_xls).data


# In[9]:


logger.debug('Building input dataframe')
cols = ['image_name', 'question', 'answer']

image_name_question = df_train[cols].copy()
image_name_question_val = df_val[cols].copy()


# ### This is just for performance and quick debug cycles! remove before actual trainining:

# In[10]:


# image_name_question = image_name_question.head(5)
# image_name_question_val = image_name_question_val.head(5)


# In[11]:


import cv2
def get_text_features(txt):
    ''' For a given txt, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    tokens = nlp(txt)    
    text_features = np.zeros((1, input_length, spacy_emmbeding_dim))
    
    num_tokens_to_take = min([input_length, len(tokens)])
    trimmed_tokens = tokens[:num_tokens_to_take]
    
    for j, token in enumerate(trimmed_tokens):
        # print(len(token.vector))
        text_features[0,j,:] = token.vector
    # Bringing to shape of (1, input_length * spacy_emmbeding_dim)
    ## ATTN - nlp vector:
    text_features = np.reshape(text_features, (1, input_length * spacy_emmbeding_dim))
    return text_features


def get_image(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''
    image_size = image_size_by_base_models[DEFAULT_IMAGE_WIEGHTS]
    im = cv2.resize(cv2.imread(image_file_name), image_size)

    # convert the image to RGBA
#     im = im.transpose((2, 0, 1))
    return im

from keras.utils import to_categorical
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


# In[12]:


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


# Note:
# This might take a while...

# In[13]:


logger.debug('----===== Preproceccing train data =====----')
image_locations = train_data.images_path
image_name_question = pre_process_raw_data(image_name_question, image_locations)


# In[14]:


logger.debug('----===== Preproceccing validation data =====----')
image_locations = validation_data.images_path
image_name_question_val = pre_process_raw_data(image_name_question_val, image_locations)


# In[15]:


image_name_question.head(2)


# #### Saving the data, so later on we don't need to compute it again

# In[16]:


from collections import namedtuple
VqaSpecs = namedtuple('VqaSpecs',['embedding_dim', 'seq_length', 'data_location','meta_data'])
def get_vqa_specs(meta_data):    
    dim = embedding_dim
    s_length = seq_length    
    return VqaSpecs(embedding_dim=dim, seq_length=s_length, data_location=data_location,meta_data=meta_data)

vqa_specs = get_vqa_specs(meta_data)

# Show waht we got...
s = str(vqa_specs)
s[:s.index('meta_data=')+10]


# In[19]:



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


# In[20]:


File.dump_pickle(vqa_specs, vqa_specs_location)
logger.debug(f"VQA Specs saved to:\n{vqa_specs_location}")

