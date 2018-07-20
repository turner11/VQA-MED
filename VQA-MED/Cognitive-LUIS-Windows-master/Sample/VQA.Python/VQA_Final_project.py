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
import re



import time, datetime
import pandas as pd
import numpy as np

import warnings

from common.utils import has_word


warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


def get_time_stamp():
    now = time.time()
    ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
    return ts


from vqa_logger import logger

# ### Collecting pre processing item

# ###### Download pre trained items & store their location

# In[2]:


# TODO: Add down loading for glove file


# In[3]:


import os
import spacy

seq_length = 26
embedding_dim = 384

vectors = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']  # 'en_vectors_web_lg'
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
ckpt_model_weights_filename = os.path.abspath('data/ckpts/model_weights.h5')

spacy_emmbeding_dim = 384

# input_dim : the vocabulary size. This is how many unique words are represented in your corpus.
# output_dim : the desired dimension of the word vector. For example, if output_dim = 100, then every word will be mapped onto a vector with 100 elements, whereas if output_dim = 300, then every word will be mapped onto a vector with 300 elements.
# input_length : the length of your sequences. For example, if your data consists of sentences, then this variable represents how many words there are in a sentence. As disparate sentences typically contain different number of words, it is usually required to pad your sequences such that all sentences are of equal length. The keras.preprocessing.pad_sequence method can be used for this (https://keras.io/preprocessing/sequence/).
input_length = 32  # longest question / answer was 28 words. Rounding up to a nice round number

# ATTN: Arbitrary  selected for both question and asnwers
embedded_sentence_length = input_length * embedding_dim
DEFAULT_IMAGE_WIEGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}

# ##### Set locations for pre-training items to-be created

# In[4]:


# Pre process results files
data_prepo_meta = os.path.abspath('data/my_data_prepro.json')
data_prepo_meta_validation = os.path.abspath('data/my_data_prepro_validation.json')
# Location of embediing pre trained matrix
embedding_matrix_filename = os.path.abspath('data/ckpts/embeddings_{0}.h5'.format(embedding_dim))

# The location to dump models to
vqa_models_folder = "C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models"

# In[5]:


from collections import namedtuple

dbg_file_csv_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA.csv'
dbg_file_xls_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA_post_pre_process_intermediate.xlsx'  # "'C:\\\\Users\\\\avitu\\\\Documents\\\\GitHub\\\\VQA-MED\\\\VQA-MED\\\\Cognitive-LUIS-Windows-master\\\\Sample\\\\VQA.Python\\\\dumped_data\\\\vqa_data.xlsx'
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

DataLocations = namedtuple('DataLocations', ['data_tag', 'raw_csv', 'raw_xls', 'processed_xls', 'images_path'])
train_data = DataLocations('train', dbg_file_csv_train, dbg_file_xls_train, dbg_file_xls_processed_train,
                           images_path_train)
validation_data = DataLocations('validation', dbg_file_csv_validation, dbg_file_xls_validation,
                                dbg_file_xls_processed_validation, images_path_validation)
test_data = DataLocations('test', dbg_file_csv_test, dbg_file_xls_test, dbg_file_xls_processed_test, images_path_test)

# Before we start, just for making sure, lets clear the session:

# In[6]:


from keras import backend as keras_backend

keras_backend.clear_session()

# ### Preprocessing and creating meta data

# We will use this function for creating meta data:

# In[7]:


from vqa_logger import logger
import itertools
import string
from utils.os_utils import File  # This is a simplehelper file of mine...


def create_meta(meta_file_location, df):
    logger.debug("Creating meta data ('{0}')".format(meta_file_location))

    def get_unique_words(col):
        single_string = " ".join(df[col])
        exclude = set(string.punctuation)
        s_no_panctuation = ''.join(ch for ch in single_string if ch not in exclude)
        unique_words = set(s_no_panctuation.split(" ")).difference({'', ' '})
        print("column {0} had {1} unique words".format(col, len(unique_words)))
        return unique_words

    cols = ['question', 'answer']
    unique_words = set(itertools.chain.from_iterable([get_unique_words(col) for col in cols]))
    print("total unique words: {0}".format(len(unique_words)))

    metadata = {}
    metadata['ix_to_word'] = {str(word): int(i) for i, word in enumerate(unique_words)}
    metadata['ix_to_ans'] = {ans: i for ans, i in enumerate(set(df['answer']))}
    # {int(i):str(word) for i, word in enumerate(unique_words)}
    print("Number of answers: {0}".format(len(set(metadata['ix_to_ans'].values()))))
    print("Number of questions: {0}".format(len(set(metadata['ix_to_ans'].values()))))

    File.dump_json(metadata, meta_file_location)
    return metadata


# And lets create meta data for training & validation sets:

# Get the data itself, Note the only things required in dataframe are:
# 1. image_name
# 2. question
# 3. answer
# 

# In[8]:


from parsers.VQA18 import Vqa18Base

df_train = Vqa18Base.get_instance(train_data.processed_xls).data
df_val = Vqa18Base.get_instance(validation_data.processed_xls).data

# In[9]:


print("----- Creating training meta -----")
meta_train = create_meta(data_prepo_meta, df_train)

print("\n----- Creating validation meta -----")
meta_validation = create_meta(data_prepo_meta, df_val)

# meta_train


# ### Creating the model

# #### The functions the gets the model:

# In[10]:


from collections import namedtuple

VqaSpecs = namedtuple('VqaSpecs', ['embedding_dim', 'seq_length', 'meta_data'])


def get_vqa_specs(meta_data):
    dim = embedding_dim
    s_length = seq_length
    return VqaSpecs(embedding_dim=dim, seq_length=s_length, meta_data=meta_data)


vqa_specs = get_vqa_specs(meta_train)

# Show waht we got...
s = str(vqa_specs)
s[:s.index('meta_data=') + 10]


# Define how to build the word-to vector branch:

# In[11]:


def word_2_vec_model(input_tensor):
    # notes:
    # num works: scalar represents size of original corpus
    # embedding_dim : dim reduction. every input string will be encoded in a binary fashion using a vector of this length
    # embedding_matrix (AKA embedding_initializers): represents a pre trained network

    LSTM_UNITS = 512
    DENSE_UNITS = 1024
    DENSE_ACTIVATION = 'relu'

    # logger.debug("Creating Embedding model")
    # x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length,trainable=False)(input_tensor)
    # x = LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(seq_length, embedding_dim))(x)
    # x = BatchNormalization()(x)
    # x = LSTM(units=LSTM_UNITS, return_sequences=False)(x)
    # x = BatchNormalization()(x)
    x = input_tensor  # Since using spacy
    x = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(x)
    model = x
    logger.debug("Done Creating Embedding model")
    return model


# In the same manner, define how to build the image representation branch:

# In[12]:


from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D  # , Input, Dropout


def get_image_model(base_model_weights=DEFAULT_IMAGE_WIEGHTS, out_put_dim=1024):
    base_model_weights = base_model_weights

    # base_model = VGG19(weights=base_model_weights,include_top=False)
    base_model = VGG19(weights=base_model_weights, include_top=False)
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D(name="image_model_average_pool")(x)
    # let's add a fully-connected layer
    x = Dense(out_put_dim, activation='relu', name="image_model_dense")(x)
    # and a logistic layer -- let's say we have 200 classes
    # predictions = Dense(200, activation='softmax')(x)
    model = x

    return base_model.input, model


# And finally, building the model itself:

# In[13]:


import keras.layers as keras_layers

# Available merge strategies:
# keras_layers.multiply, keras_layers.add, keras_layers.concatenate, 
# keras_layers.average, keras_layers.co, keras_layers.dot, keras_layers.maximum

merge_strategy = keras_layers.concatenate

# In[14]:


from keras import Model, models, Input, callbacks
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, Embedding, LSTM, BatchNormalization  # , GlobalAveragePooling2D, Merge, Flatten


def get_vqa_model(meta):
    #     import tensorflow as tf
    #     g = tf.Graph()
    #     with g.as_default():
    DENSE_UNITS = 1000
    DENSE_ACTIVATION = 'relu'

    OPTIMIZER = 'rmsprop'
    LOSS = 'categorical_crossentropy'
    METRICS = 'accuracy'
    num_classes = len(meta['ix_to_ans'].keys())
    image_model, lstm_model, fc_model = None, None, None
    try:
        # ATTN:
        lstm_input_tensor = Input(shape=(embedded_sentence_length,), name='embedding_input')
        # lstm_input_tensor = Input(shape=(embedding_dim,), name='embedding_input')

        logger.debug("Getting embedding (lstm model)")
        lstm_model = word_2_vec_model(input_tensor=lstm_input_tensor)

        logger.debug("Getting image model")
        out_put_dim = lstm_model.shape[-1].value
        image_input_tensor, image_model = get_image_model(out_put_dim=out_put_dim)

        logger.debug("merging final model")
        fc_tensors = merge_strategy(inputs=[image_model, lstm_model])
        fc_tensors = BatchNormalization()(fc_tensors)
        fc_tensors = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(fc_tensors)
        fc_tensors = BatchNormalization()(fc_tensors)

        # ATTN:
        fc_tensors = Dense(units=embedded_sentence_length, activation='softmax', name='model_output_sofmax_dense')(
            fc_tensors)
        # fc_tensors = Dense(units=num_classes, activation='softmax', name='model_output_sofmax_dense')(fc_tensors)

        fc_model = Model(inputs=[lstm_input_tensor, image_input_tensor], output=fc_tensors)
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


model = get_vqa_model(meta_train)
model

# And the summary of our model:

# In[15]:


## If you are getting errors about installing pydot, add the path of dot.exe to PATH:
# sys.path.append('PATH_TO_DOT_EXE')

import graphviz
import pydot
from keras.utils import plot_model
# plot_model(model, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

model_to_dot(model)
# SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[16]:


model.summary()

# We better save it:

# In[17]:


import graphviz
import pydot
from keras.utils import plot_model


def print_model_summary_to_file(fn, model):
    # Open the file
    with open(fn, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


ts = get_time_stamp()

now_folder = os.path.abspath('{0}\\{1}\\'.format(vqa_models_folder, ts))
model_fn = os.path.join(now_folder, 'vqa_model.h5')
model_image_fn = os.path.join(now_folder, 'model_vqa.png5')
summary_fn = os.path.join(now_folder, 'model_summary.txt')
logger.debug("saving model to: '{0}'".format(model_fn))

fn_image = os.path.join(now_folder, 'model.png')
logger.debug(f"saving model image to {fn_image}")

try:
    File.validate_dir_exists(now_folder)
    model.save(model_fn)  # creates a HDF5 file 'my_model.h5'
    logger.debug("model saved")
except Exception as ex:
    logger.error("Failed to save model:\n{0}".format(ex))

try:
    logger.debug("Writing Symmary")
    print_model_summary_to_file(summary_fn, model)
    logger.debug("Done Writing Summary")

    logger.debug("Saving image")
    plot_model(model, to_file=fn_image)
    logger.debug(f"Image saved ('{fn_image}')")
#     logger.debug("Plotting model")
#     plot_model(model, to_file=model_image_fn)
#     logger.debug("Done Plotting")
except Exception as ex:
    logger.warning("{0}".format(ex))

# In[18]:


# %matplotlib inline
# from matplotlib import pyplot as plt
# %pylab inline


# plt.imshow(img,cmap='gray')
# plt.show()

from IPython.display import Image, display

listOfImageNames = [fn_image]

for imageName in listOfImageNames:
    display(Image(filename=imageName))

# ### Training the model

# In[19]:


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
        text_features[0, j, :] = token.vector
    # Bringing to shape of (1, input_length * spacy_emmbeding_dim)
    ## ATTN:
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


# from keras.utils import to_categorical
# def get_categorial_labels(df, meta):
#     classes = df['answer']
#     class_count = len(classes)
#     classes_indices = list(meta['ix_to_ans'].keys())
#     categorial_labels = to_categorical(classes_indices, num_classes=class_count)

#     return categorial_labels

# categorial_labels_train = get_categorial_labels(df_train, meta_train)
# categorial_labels_val = get_categorial_labels(df_val, meta_validation)


# ### Preparing the data for training

# Note:
# This might take a while...

# In[20]:


logger.debug('Building input dataframe')
cols = ['image_name', 'question', 'answer']

image_name_question = df_train[cols].copy()
image_name_question_val = df_val[cols].copy()


# del df_train
# del df_val


# ### This is just for performance and quick debug cycles! remove before actual trainining:

# In[21]:


# image_name_question = image_name_question.head(5)
# image_name_question_val = image_name_question_val.head(5)


# In[22]:


def pre_process_raw_data(df, images_path):
    df['image_name'] = df['image_name'].apply(lambda q: q if q.lower().endswith('.jpg') else q + '.jpg')

    df['path'] = df['image_name'].apply(lambda name: os.path.join(images_path, name))

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




# In[23]:


logger.debug('Preproceccing train data')
image_locations = train_data.images_path
image_name_question = pre_process_raw_data(image_name_question, image_locations)

# In[24]:


logger.debug('Preproceccing validation data')
image_locations = validation_data.images_path
image_name_question_val = pre_process_raw_data(image_name_question_val, image_locations)


# #### Saving the data, so later on we don't need to compute it again

# In[25]:


# logger.debug("Save the data")

# item_to_save = image_name_question
# item_to_save = image_name_question.head(10)

# item_to_save.to_hdf('model_input.h5', key='df')    
# # store = HDFStore('model_input.h5')
# logger.debug("Saved")


# #### Loading the data after saved:

# In[26]:


# if image_name_question is None:
#     logger.debug("Load the data")
#     from pandas import HDFStore
#     store = HDFStore('model_input.h5')
#     image_name_question = store['df']  

# logger.debug(f"Shape: {image_name_question.shape}")
# image_name_question.head(2)


# #### Packaging the data to be in expected input shape

# In[27]:


def concate_row(df, col):
    return np.concatenate(df[col], axis=0)


def get_features_and_labels(df):
    image_features = np.asarray([np.array(im) for im in df['image']])
    # np.concatenate(image_features['question_embedding'], axis=0).shape
    question_features = concate_row(df, 'question_embedding')

    features = ([f for f in [question_features, image_features]])
    labels = concate_row(df, 'answer_embedding')
    return features, labels


features_t, labels_t = get_features_and_labels(image_name_question)
features_val, labels_val = get_features_and_labels(image_name_question_val)
validation_input = (features_val, labels_val)

# features_val
# image_name_question.head(2)
# image_name_question_val.head(2)
# Note: The shape of answer (for a single recored ) is (number of words, 384)


# In[28]:


# model.input_layers
# model.input_layers_node_indices
# model.input_layers_tensor_indices
# model.input_mask
# model.input_names


# model.inputs
# model.input
# model.input_spec

# print(f'Wrapper shape:{train_features.shape}')
# model.input_shape, np.concatenate(train_features).shape
# model.input_shape, train_features[0].shape,  train_features[1].shape

print(f'Expectedt shape: {model.input_shape}')
print('---------------------------------------------------------------------------')
print(f'Actual training shape:{features_t[0].shape, features_t[1].shape}')
print(f'Train Labels shape:{labels_t.shape}')
print('---------------------------------------------------------------------------')
print(f'Actual Validation shape:{features_val[0].shape, features_val[1].shape}')
print(f'Validation Labels shape:{labels_val.shape}')

# #### Performaing the actual training

# In[29]:


from keras.utils import plot_model

# train_features = image_name_question
# validation_input = (validation_features, categorial_validation_labels)

## construct the image generator for data augmentation
# aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#                                horizontal_flip=True, fill_mode="nearest")
# train_generator = aug.flow(train_features, categorial_train_labels)

# stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,mode='auto')

try:
    #     history = model.fit_generator(train_generator,
    #                                   validation_data=validation_input,
    #                                   steps_per_epoch=len(train_features) // self.batch_size,
    #                                   epochs=self.epochs,
    #                                   verbose=1,
    #                                   callbacks=[stop_callback],
    #                                   class_weight=class_weight
    #                                   )
    # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    history = model.fit(features_t, labels_t,
                        # epochs=epochs,
                        # batch_size=batch_size,
                        validation_data=validation_input)
except Exception as ex:
    logger.error("Got an error training model: {0}".format(ex))
    #     model.summary(print_fn=logger.error)
    raise
# return model, history


# In[30]:


features_t
labels_t
# features_val
# labels_val 
# validation_input
