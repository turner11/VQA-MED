
# coding: utf-8

# # VQA - MED

# This notebook demonstrates the efforts made for Visual Q&A based on the data set from VQA-Med 2018 contest

# ## Abstract

# The inputs for VQA are:
# 1. The question text 
# 2. The image
# 
# The question text is being embedded into a feature vector using a pre-traing [globe file](https://nlp.stanford.edu/projects/glove/). 
# 
# In a similar manner the image is being processed using a pre trained deep NN (e.g. [VGG](http://qr.ae/TUTEKo) with initial wights of a pretrained [imagenet model](https://en.wikipedia.org/wiki/ImageNet))
# 

# ## The plan

# 0. [Preperations and helpers](#Preperations-and-helpers)
# 1. [Collecting pre processing item](#Collecting-pre-processing-item)
# 2. [Preprocessing and creating meta data](#Preprocessing-and-creating-meta-data)
# 3. [Creating the model](#Creating-the-model)
# 4. [Training the model](#Training-the-model)
# 5. [Testing the model](#Testing-the-model)
# 

# ### Preperations and helpers

# The following are just helpers & utils imports - feel free to skip...

# In[1]:
from pandas import HDFStore

from parsers.utils import VerboseTimer
from utils.os_utils import File, print_progress
import time, datetime

def get_time_stamp():
    now = time.time()
    ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
    return ts


# ### Collecting pre processing item

# ###### Download pre trained items & store their location

# In[2]:

#TODO: Add down loading for glove file


# In[3]:

import os
seq_length =    26
embedding_dim = 300

glove_path =                    os.path.abspath('data/glove.6B.{0}d.txt'.format(embedding_dim))
embedding_matrix_filename =     os.path.abspath('data/ckpts/embeddings_{0}.h5'.format(embedding_dim))
ckpt_model_weights_filename =   os.path.abspath('data/ckpts/model_weights.h5')




DEFAULT_IMAGE_WIEGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}


# In[4]:

import os
# Fail fast...
suffix = "Failing fast:\n"
assert os.path.isfile(glove_path), suffix+"glove file does not exists:\n{0}".format(glove_path)
# assert os.path.isfile(embedding_matrix_filename), suffix+"Embedding matrix file does not exist:\n{0}".format(embedding_matrix_filename)
assert os.path.isfile(ckpt_model_weights_filename), suffix+"glove file does not exists:\n{0}".format(ckpt_model_weights_filename)

print('Validated file locations')


# ##### Set locations for pre-training items to-be created

# In[5]:

# Pre process results files
data_prepo_meta            = os.path.abspath('data/my_data_prepro.json')
data_prepo_meta_validation = os.path.abspath('data/my_data_prepro_validation.json')
# Location of embediing pre trained matrix
embedding_matrix_filename  = os.path.abspath('data/ckpts/embeddings_{0}.h5'.format(embedding_dim))

# The location to dump models to
vqa_models_folder          = "C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models"



# ### Preprocessing and creating meta data

# We will use this function for creating meta data:

# In[6]:

from vqa_logger import logger 
import itertools
import string
from utils.os_utils import File #This is a simplehelper file of mine...

def create_meta(meta_file_location, df):
        logger.debug("Creating meta data ('{0}')".format(meta_file_location))
        def get_unique_words(col):
            single_string = " ".join(df[col])
            exclude = set(string.punctuation)
            s_no_panctuation = ''.join(ch for ch in single_string if ch not in exclude)
            unique_words = set(s_no_panctuation.split(" ")).difference({'',' '})
            print("column {0} had {1} unique words".format(col,len(unique_words)))
            return unique_words

        cols = ['question', 'answer']
        unique_words = set(itertools.chain.from_iterable([get_unique_words(col) for col in cols]))
        print("total unique words: {0}".format(len(unique_words)))

        metadata = {}
        metadata['ix_to_word'] = {str(word): int(i) for i, word in enumerate(unique_words)}
        metadata['ix_to_ans'] = {ans:i for ans, i in enumerate(set(df['answer']))}
        # {int(i):str(word) for i, word in enumerate(unique_words)}

        File.dump_json(metadata,meta_file_location)
        return metadata


# And lets create meta data for training & validation sets:

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


# Get the data itself, Note the only things required in dataframe are:
# 1. image_name
# 2. question
# 3. answer
# 

# In[8]:

from parsers.VQA18 import Vqa18Base
df_train = Vqa18Base.get_instance(train_data.processed_xls).data            
df_val = Vqa18Base.get_instance(validation_data.processed_xls).data
# df_train.head(2)


# In[9]:

print("----- Creating training meta -----")
meta_train = create_meta(data_prepo_meta, df_train)

print("\n----- Creating validation meta -----")
meta_validation = create_meta(data_prepo_meta, df_val)

# meta_train


# ### Creating the model

# #### The functions the gets the model:

# ##### Get Embedding:

# In[10]:

import numpy as np
import random
import h5py
def prepare_embeddings(metadata):
    embedding_filename = embedding_matrix_filename
    num_words = len(metadata['ix_to_word'].keys())
    dim_embedding = embedding_dim



    logger.debug("Embedding Data...")
    # texts = df['question']

    embeddings_index = {}
    i = -1
    line = "NO DATA"


    glove_line_count = File.file_len(glove_path, encoding="utf8")
    def process_line(i, line):
        print_progress(i, glove_line_count)
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            print_progress(i+1, glove_line_count)
        except Exception as ex:
            logger.error(
                "An error occurred while working on glove file [line {0}]:\n"
                "Line text:\t{1}\nGlove path:\t{2}\n"
                "{3}".format(
                    i, line, glove_path, ex))
            raise


    # with open(glove_path, 'r') as glove_file:
    with VerboseTimer("Embedding"):
        with open(glove_path, 'r', encoding="utf8") as glove_file:
            [process_line(i=i, line=line)for i, line in enumerate(glove_file)]



    embedding_matrix = np.zeros((num_words, dim_embedding))
    word_index = metadata['ix_to_word']

    with VerboseTimer("Creating matrix"):
        embedding_tupl = ((word, i, embeddings_index.get(word)) for word, i in word_index.items())
        embedded_with_values = [(word, i, embedding_vector) for word, i, embedding_vector in embedding_tupl if embedding_vector is not None]

        for word, i, embedding_vector in embedded_with_values:
            embedding_matrix[i] = embedding_vector


    e = {tpl[0] for tpl in embedded_with_values}
    w = set(word_index.keys())
    words_with_no_embedding = w-e
    rnd = random.sample(words_with_no_embedding , 5)
    logger.debug("{0} words did not have embedding. e.g.:\n{1}".format(len(words_with_no_embedding),rnd))

    with VerboseTimer("Dumping matrix"):
        with h5py.File(embedding_filename, 'w') as f:
            f.create_dataset('embedding_matrix', data=embedding_matrix)

    return embedding_matrix



# If the embedding already exists, save yourself the time and just load it.  
# Otherwise - calculate it

# In[11]:

if os.path.exists(embedding_matrix_filename):
    logger.debug("Embedding Data already exists. Loading...")
    with h5py.File(embedding_matrix_filename) as f:
        embedding_train = np.array(f['embedding_matrix'])    
else:
    logger.debug("Calculating Embedding...")
    embedding_train = prepare_embeddings(meta_train)
    
embedding_matrix = embedding_train


# And lets take a look:

# In[12]:

embedding_matrix


# And lets wrap it with related information:

# In[13]:

from vqa_flow.data_structures import EmbeddingData
def get_embedding_data(embedding_matrix, meta_data):    
    dim = embedding_dim
    s_length = seq_length    
    return EmbeddingData(embedding_matrix=embedding_matrix,embedding_dim=dim, seq_length=s_length, meta_data=meta_data)

embedding_train = get_embedding_data(embedding_matrix, meta_train)
str(embedding_train)


# Define how to build the word-to vector branch:

# In[14]:

def word_2_vec_model(embedding_matrix, num_words, embedding_dim, seq_length, input_tensor):
        # notes:
        # num works: scalar represents size of original corpus
        # embedding_dim : dim reduction. every input string will be encoded in a binary fashion using a vector of this length
        # embedding_matrix (AKA embedding_initializers): represents a pre trained network

        LSTM_UNITS = 512
        DENSE_UNITS = 1024
        DENSE_ACTIVATION = 'relu'


        logger.debug("Creating Embedding model")
        x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length,trainable=False)(input_tensor)
        x = LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(seq_length, embedding_dim))(x)
        x = BatchNormalization()(x)
        x = LSTM(units=LSTM_UNITS, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(x)
        model = x
        logger.debug("Done Creating Embedding model")
        return model


# In the same manner, define how to build the image representation branch:

# In[15]:

from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D#, Input, Dropout
def get_image_model(base_model_weights=DEFAULT_IMAGE_WIEGHTS, out_put_dim=1024):
    base_model_weights = base_model_weights

    # base_model = VGG19(weights=base_model_weights,include_top=False)
    base_model = VGG19(weights=base_model_weights, include_top=False)
    base_model.trainable = False

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D(name="image_model_average_pool")(x)
    # let's add a fully-connected layer
    x = Dense(out_put_dim, activation='relu',name="image_model_dense")(x)
    # and a logistic layer -- let's say we have 200 classes
    # predictions = Dense(200, activation='softmax')(x)
    model = x
    
    return base_model.input , model


# Before we start, just for making sure, lets clear the session:

# In[16]:

from keras import backend as keras_backend
keras_backend.clear_session()


# And finally, building the model itself:

# In[17]:

import keras.layers as keras_layers
#Available merge strategies:
# keras_layers.multiply, keras_layers.add, keras_layers.concatenate, 
# keras_layers.average, keras_layers.co, keras_layers.dot, keras_layers.maximum
            
merge_strategy = keras_layers.concatenate


# In[18]:

from keras import Model, models, Input, callbacks
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, Embedding, LSTM, BatchNormalization#, GlobalAveragePooling2D, Merge, Flatten

def get_vqa_model(embedding_data=None):        
        embedding_matrix = embedding_data.embedding_matrix
        num_words = embedding_data.num_words
        num_classes = embedding_data.num_classes

        DENSE_UNITS = 1000
        DENSE_ACTIVATION = 'relu'

        OPTIMIZER = 'rmsprop'
        LOSS = 'categorical_crossentropy'
        METRICS = 'accuracy'

        image_model, lstm_model, fc_model = None, None, None
        try:

            lstm_input_tensor = Input(shape=(embedding_dim,), name='embedding_input')

            logger.debug("Getting embedding (lstm model)")
            lstm_model = word_2_vec_model(embedding_matrix=embedding_matrix, num_words=num_words, embedding_dim=embedding_dim,
                                               seq_length=seq_length, input_tensor=lstm_input_tensor)

            logger.debug("Getting image model")
            out_put_dim = lstm_model.shape[-1].value
            image_input_tensor, image_model = get_image_model(out_put_dim=out_put_dim)


            logger.debug("merging final model")
            fc_tensors = merge_strategy(inputs=[image_model, lstm_model])
            fc_tensors = BatchNormalization()(fc_tensors)
            fc_tensors = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(fc_tensors)
            fc_tensors = BatchNormalization()(fc_tensors)
            fc_tensors = Dense(units=num_classes, activation='softmax')(fc_tensors)

            fc_model = Model(input=[lstm_input_tensor, image_input_tensor], output=fc_tensors)
            fc_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])
        except Exception as ex:
            logger.error("Got an error while building vqa model:\n{0}".format(ex))
            models = [(image_model, 'image_model'), (lstm_model, 'lstm_model'), (fc_model, 'lstm_model')]
            for m, name in models:
                if m is not None:
                    logger.error("######################### {0} model details: ######################### ".format(name))
                    try:
                        m.summary(print_fn=logger.error)
                    except Exception as ex2:
                        logger.warning("Failed to print summary for {0}:\n{1}".format(name, ex2))
            raise

        return fc_model

model = get_vqa_model(embedding_data=embedding_train)
model


# And the summary of our model:

# In[19]:

model.summary()


# We better save it:

# In[20]:

def print_model_summary_to_file(fn, model):
    # Open the file
    with open(fn,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        

ts = get_time_stamp()

now_folder = os.path.abspath('{0}\\{1}\\'.format(vqa_models_folder, ts))
model_fn = os.path.join(now_folder, 'vqa_model.h5')
model_image_fn = os.path.join(now_folder, 'model_vqa.png5')
summary_fn = os.path.join(now_folder, 'model_summary.txt')
logger.debug("saving model to: '{0}'".format(model_fn))

try:
    File.validate_dir_exists(now_folder)
    model.save(model_fn)  # creates a HDF5 file 'my_model.h5'
    logger.debug("model saved")
except Exception as ex:
    logger.error("Failed to save model:\n{0}".format(ex))

try:
    logger.debug("Writing history")
    print_model_summary_to_file(summary_fn, model)
    logger.debug("Done Writing History")
#     logger.debug("Plotting model")
#     plot_model(model, to_file=model_image_fn)
#     logger.debug("Done Plotting")
except Exception as ex:
    logger.warning("{0}".format(ex))


# ### Training the model

# In[21]:

print(model)

# ------------------------------------------------------------------------------------------------------------------------
from keras.utils import plot_model, to_categorical
import cv2
keras_backend.clear_session()

def get_categorial_labels(df, meta):
    classes = df['answer']
    class_count = len(classes)
    classes_indices = list(meta['ix_to_ans'].keys())
    categorial_labels = to_categorical(classes_indices, num_classes=class_count)

    return categorial_labels

categorial_labels_train = get_categorial_labels(df_train, meta_train)
categorial_labels_val = get_categorial_labels(df_val, meta_validation)



image_model = get_image_model()
def get_image(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''
    image_size = image_size_by_base_models[DEFAULT_IMAGE_WIEGHTS]
    im = cv2.resize(cv2.imread(image_file_name), image_size)

    # convert the image to RGBA
    im = im.transpose((2, 0, 1))
    return im


import spacy
vectors = ['en_core_web_lg','en_core_web_md', 'en_core_web_sm']#'en_vectors_web_lg'
vector = vectors[2]
print(f'using vector: {vector}')
#
# en_core_web_md, en_core_web_lg, en_vectors_web_lg
# en_core_web_sm
# 'en_glove_cc_300_1m_vectors'

word_embeddings = spacy.load('en', vectors=vector)
def get_question_features(question):
    ''' For a given question, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, len(tokens), 384))

    for j, token in enumerate(tokens):
        # print(len(token.vector))
        question_tensor[0,j,:] = token.vector

    return question_tensor


image_name_question = df_train[['image_name', 'question']].copy()
image_name_question['image_name'] = image_name_question['image_name']\
                                    .apply(lambda q: q if q.lower().endswith('.jpg') else q+'.jpg')

image_name_question['path'] =  image_name_question['image_name']\
                                .apply(lambda name:os.path.join(train_data.images_path, name))

existing_files = [os.path.join(train_data.images_path, fn) for fn in os.listdir(train_data.images_path)]
image_name_question = image_name_question.loc[image_name_question['path'].isin(existing_files)]

image_name_question['image'] = image_name_question['path']\
                                .apply(lambda im_path: get_image(im_path))

image_name_question['question_embedding'] = image_name_question['question']\
                                            .apply(lambda q: get_question_features(q))
image_name_question.to_hdf('model_input.h5', key='df')
store = HDFStore('model_input.h5')

store['df'] = image_name_question  # save it
store['df']  # load it
images_path = [(fn, os.path.join(train_data.images_path, fn)) for fn in os.listdir(train_data.images_path) if fn.endswith('jpg')]
image_names = [t[0] for t in images_path]






df_train['image_path'] = df_train
questions = 1
image_file_name = images_path[0]
images = np.array([get_image(image_fn) for image_fn in images_path])

image_model[-1].predict(images)


get_image(image_file_name)
image_features = [get_image(image_fn) for image_fn in images_path]












# validation_data = (validation_features, categorial_validation_labels)
#
# ## construct the image generator for data augmentation
# # aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
# #                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# #                                horizontal_flip=True, fill_mode="nearest")
# # train_generator = aug.flow(train_features, categorial_train_labels)
#
# stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,mode='auto')
#
# try:
#     history = model.fit_generator(train_generator,
#                                   validation_data=validation_data,
#                                   steps_per_epoch=len(train_features) // self.batch_size,
#                                   epochs=self.epochs,
#                                   verbose=1,
#                                   callbacks=[stop_callback],
#                                   class_weight=class_weight
#                                   )
#     # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
#
#     # history = model.fit(train_features,train_labels,
#     #                     epochs=epochs,
#     #                     batch_size=batch_size,
#     #                     validation_data=validation_data)
# except Exception as ex:
#     logger.error("Got an error training model: {0}".format(ex))
#     model.summary(print_fn=logger.error)
#     raise
# return model, history