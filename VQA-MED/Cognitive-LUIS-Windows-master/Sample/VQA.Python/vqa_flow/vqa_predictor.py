import os
import random
import string

import datetime
from functools import partial

import pandas as pd
import json
import h5py
import numpy as np
import time

from keras import Model
from keras.utils import plot_model

from parsers.utils import VerboseTimer
from pre_processing.known_find_and_replace_items import vqa_models_folder
from utils.os_utils import File, print_progress
from vqa_flow.constatns import embedding_matrix_filename, data_prepo_meta, embedding_dim, glove_path, seq_length
from vqa_flow.data_structures import EmbeddingData
from vqa_flow.image_models import ImageModelGenerator
from vqa_flow.keras_utils import print_model_summary_to_file
from vqa_logger import logger
from keras.layers import Dense, Dropout, Embedding, LSTM, Merge, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential #Model
import itertools

class VqaPredictor(object):
    """"""

    def __init__(self, embedding_matrix_path=None, image_model_initial_weights=None, merge_strategy='concat'):
        """"""
        assert callable(merge_strategy) or merge_strategy in ['mul','sum', 'concat', 'ave', 'cos', 'dot', 'max']

        super(VqaPredictor, self).__init__()
        self.image_model_initial_weights = image_model_initial_weights
        self.merge_strategy = merge_strategy
        self.embedding_matrix_path = embedding_matrix_path or embedding_matrix_filename

    def __repr__(self):
        return super(VqaPredictor, self).__repr__()

    def word_2_vec_model(self, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate=0.5):
        # notes:
        # num works: scalar represents size of original corpus
        # embedding_dim : dim reduction. every input string will be encoded in a binary fashion using a vector of this length
        # embedding_matrix (AKA embedding_initializers): represents a pre trained network

        LSTM_UNITS = 512
        DENSE_UNITS = 1024
        DENSE_ACTIVATION = 'relu'


        logger.debug("Creating Text model")
        model = Sequential()

        layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length,
                          trainable=False)
        model.add(layer)

        lstm1 = LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(seq_length, embedding_dim))
        model.add(lstm1)

        norm_barch1 = BatchNormalization()
        model.add(norm_barch1)

        lstm2 = LSTM(units=LSTM_UNITS, return_sequences=False)
        model.add(lstm2)

        norm_batch2 = BatchNormalization()
        model.add(norm_batch2)

        dense = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)
        model.add(dense)
        return model

    #
    # def img_model(self, dropout_rate):
    #     DENSE_UNITS = 1024
    #     INPUT_DIM = 4096
    #     DENSE_ACTIVATION = 'relu'
    #     logger.debug("Creating image model")
    #     model = Sequential()
    #     dense = Dense(units=DENSE_UNITS, input_dim=INPUT_DIM, activation=DENSE_ACTIVATION)
    #     model.add(dense)
    #     return model
    #
    def get_vqa_model(self, embedding_data=None):
        embedding_data = embedding_data or VqaPredictor.get_embedding_data()
        embedding_matrix = embedding_data.embedding_matrix
        num_words = embedding_data.num_words
        num_classes = embedding_data.num_classes




        DROPOUT_RATE = 0.5
        DENSE_UNITS = 1000
        DENSE_ACTIVATION = 'relu'

        OPTIMIZER = 'rmsprop'
        LOSS = 'categorical_crossentropy'
        METRICS = 'accuracy'

        image_model, lstm_model, fc_model = None, None, None
        try:



            logger.debug("Getting embedding (lstm model)")
            lstm_model = self.word_2_vec_model(embedding_matrix=embedding_matrix, num_words=num_words, embedding_dim=embedding_dim,
                                               seq_length=seq_length, dropout_rate=DROPOUT_RATE)

            logger.debug("Getting image model")
            out_put_dim = lstm_model.output_shape[-1]
            image_model = ImageModelGenerator.get_image_model(self.image_model_initial_weights, out_put_dim=out_put_dim)

            fc_model = Sequential()

            logger.debug("merging final model")
            merge = Merge([image_model, lstm_model], mode=self.merge_strategy)
            fc_model.add(merge)

            # batch_norm1 = BatchNormalization()
            # fc_model.add(batch_norm1)

            dense1 = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)
            fc_model.add(dense1)

            # batch_norm2 = BatchNormalization(DROPOUT_RATE)
            # fc_model.add(batch_norm2)

            dense2 = Dense(units=num_classes, activation='softmax')
            fc_model.add(dense2)

            fc_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])
        except Exception as ex:
            logger.error("Got an error while building vqa model:\n{0}".format(ex))
            models = [(image_model, 'image_model'), (lstm_model, 'lstm_model'), (fc_model, 'lstm_model')]
            for m, name in models:
                if m is not None:
                    logger.error("######################### {0} model details: ######################### ".format(name))
                    m.summary(print_fn=logger.error)
            raise

        self.got_model(fc_model)
        return fc_model
    #
    # # segmentation: {'counts': 'j19[6h1ZNXNf1h1ZNYNe1g1[NYNf1f1YNZNh1e1YN^Nd1b1ZNbNd1^1ZNdNf1\\1ZNdNf1\\1ZNdNlKLo4_1UOBj0>kNNRLUNi4l1UO0RLUNh4k1UO;l0DTO<R1^OnNb0R1^OnNc0T1ZOlNg0X1TOhNm0V1SOkNm0T1TOlNl0T1TOjNbNmKZ2X5UOkNm0V1ROkNm0T1TOlNk0U1UOlNj0T1UOmNk0S1UOmNk0S1UOmNk0S1UOmNl0R1TOoNk0l0gMmJ01^1W4j0j0kMoJ[1W4j0j0kMoJ[1X4i0a0UNVKQ1Z4j0`0D@<`0D@<`0D@<`0D@<`0D@<`0D@=?CA=?BB>=CC==CB>:FF:?A@`0a0_O_Oa0b0^O_Ob0`0^O_Oc0a0]O@b0`0]OCa0<@D`0;AE?:BG=8DI;7EI<6DK;1I06OJ35MK35MK35MK35MK35MK45KK55KK55KK55KK64JK75IK75IK75IK75IK75IK84HL85GJ:6FJ:6FJ;5EK;5EK;5EK;5EK;5EK:7EI;7EH;9EG::FF9;GE9;GE9;GE9<EE:<FD:<FC;=EC=;CE=;CE<<DD<<DD=;CE=<BD?WNWKM5k1U41P1JPO6P1JoN7Q1IoN7Q1IoN7R1HnN8Q1JnN5S1KmN5S1KmN5S1KmN5T1JlN6T1JlN6T1JlN6S1KlN6P1SNkJg1U45n0WNmJe1T44o0XNlJd1U44o0XNkJe1V43o03QOMm0TNmJo1V4Mj08VOHi0:VOFk09UOGk09UOGk09UOFl0:SOGn08ROHn08ROHn08ROHn08UOEk0;WOCk0;VODk0<TOWNgKR1Y5c0ROYNeKT1Y5c0oNZNjKS1W5c0nNZNUMJn3k1nNZNb2f1aMWN_2j1aMSNeKOk6m1`40000002N1O2N01N10000O010O10002N001O000010O0000O10001M2L`D\\Nb;c1301VOZDMh;L_DEP<J`me33\\SZL5L4K5L4L5G8J8I4fNaNcF_1[9cNeF]1[9cNbF`1]9cNaELl0f1^9_NfELk0e1_9cNbF\\1^9dNaF]1_9dN`F\\1`9dN`F\\1`9dN`F\\1`9eN_F[1U8UORGTO<<=Z1Q8ZNVGQ1:]O?X1o7BaGTO`0Z1j7`0TH@i7R3000000000O100000001O0O2I7eNZ1ZNf10001O000001O01O00O11O3N0O000MRDkNm;U1TDjNm;U14O101M_Eb0Z7]OgHb0[7]OeHc0[7]OeHc0[7^OdHb0\\7^OdHb0]7]OcHc0]7]OcHc0]7]OcHb0b7YO_Hg0c7UO_Hl0a7UO]Hj0k7oNUHQ1l7nNTHQ1m7kNWHU1o900000O1O01000O10_NZD^10aNg;a10000O10000000001O000`dU1', 'size': [426, 640]}
    # # area: 25483.0
    # # iscrowd: 0
    # # image_id: 139
    # # bbox: [0.0, 38.0, 549.0, 297.0]
    # # category_id: 98
    # # id: 20000000
    #
    # def try_run_lstm_model(self, lstm_model, seq_length):
    #     set_1 = np.array(["Hello World!", "To infinite and beyond!", "so, we meet again mister Bond",
    #                       "To boldly go where no one has gone before",
    #                       "This list length has {seq_length} items"])
    #
    #     set_2 = np.array(["Each", "word", "is", "an", "item"])
    #
    #     set_3 = np.array([1, 2, 3, 4, 5])
    #
    #     res = defaultdict(lambda: [])
    #     template = "{0}:\n\t\t{1}"
    #     for s in [set_1, set_2, set_3]:
    #         try:
    #             assert len(s) == seq_length, "got an unexpected set length {0}".format(len(s))
    #             p = lstm_model.predict(s)
    #             res["SUCCESS"].append(template.format(s, p))
    #         except Exception as ex:
    #             res["Failed"].append(template.format(s, ex))
    #
    #     return res
    #
    # def images_to_embedded(self, images_path):
    #     COL_IMAGE = 'image'
    #     COL_PATH = 'full_path'
    #     COL_EMBEDDED = 'embedded'
    #     images_files = [os.path.join(images_path, fn) for fn in os.listdir(images_path) if fn.lower().endswith('jpg')]
    #     df = pd.DataFrame(columns=[COL_IMAGE, COL_PATH, COL_EMBEDDED])
    #     df.set_index(COL_IMAGE)
    #     df[COL_PATH] = images_files
    #     image_from_path = lambda path: os.path.split(path)[1]
    #     df[COL_IMAGE] = df[COL_PATH].apply(image_from_path)
    #     # # This causes out of memory
    #     base_model = VGG19(weights='imagenet')
    #     length_df = len(df)
    #
    #     global embeded_idx
    #     embeded_idx = 0
    #
    #     def get_embedded(fn):
    #         global embeded_idx
    #         logger.debug("embedded {0}/{1}".format(embeded_idx + 1, length_df))
    #         embeded_idx += 1
    #         try:
    #             features = Image2Features(fn, base_model=base_model)
    #         except Exception as ex:
    #             features = None
    #             logger.warning("Failed on embedded {0}:\n{1}".format(embeded_idx + 1, ex))
    #         return features
    #
    #     df['embedded'] = df['full_path'].apply(get_embedded)
    #     # length_df = len(df)
    #     # logger.debug("getting embedded for {0} images".format(length_df))
    #     # base_model = VGG19(weights='imagenet')
    #     # for i, row in df.iterrows():
    #     #     try:
    #     #         logger.debug("embedded {0}/{1}".format(i + 1, length_df))
    #     #         image_full_path = df.get_value(i, COL_PATH)
    #     #         embedded = Image2Features(image_full_path, base_model=base_model )
    #     #     except Exception as ex:
    #     #         embedded = None
    #     #         logger.warning("Failed on embedded {0}:\n{1}".format(i + 1, ex))
    #     #     df.set_value(i, COL_EMBEDDED, embedded)
    #     dump_path = os.path.join(images_path, 'embbeded_images.hdf')
    #     df.to_hdf(dump_path, key='image')
    #     return dump_path
    #
    #
    #
    #
    #
    #
    #
    #
    # def train_all_tags(self):
    #
    #
    #     for tag in all_tags:
    #         try:
    #             logger.debug("Train tag {0}.".format(tag))
    #             model, history = train_tag(tag)
    #             logger.debug("Done train tag {0}.".format(tag))
    #
    #             model_fn = os.path.abspath('{0}\\{1}_model.h5'.format(models_folder, tag))
    #             logger.debug("saving model for {0} to: '{1}'".format(tag, model_fn))
    #             model.save(model_fn)  # creates a HDF5 file 'my_model.h5'
    #             logger.debug("model saved")
    #         except Exception as ex:
    #             logger.warning("Failed to train tag {0}:\n{1}".format(tag, ex))

    # def Image2Features(image_path, base_model=None):
    #     model = get_image_model(base_model)
    #     x = image_to_preprocess_input(image_path)
    #     block4_pool_features = model.predict(x)
    #     return block4_pool_features


    @staticmethod
    def create_meta(meta_file_location, df):
        logger.debug("Creating meta data ('{0}')".format(meta_file_location))
        def get_unique_words(col):
            single_string = " ".join(df[col])
            exclude = set(string.punctuation)
            s_no_panctuation = ''.join(ch for ch in single_string if ch not in exclude)
            unique_words = set(s_no_panctuation.split(" ")).difference({'',' '})
            print("{0}: {1}".format(col,len(unique_words)))
            return unique_words

        cols = ['question', 'answer']
        unique_words = set(itertools.chain.from_iterable([get_unique_words(col) for col in cols]))
        print("total unique words: {0}".format(len(unique_words)))

        metadata = {}
        metadata['ix_to_word'] = {str(word): int(i) for i, word in enumerate(unique_words)}
        # {int(i):str(word) for i, word in enumerate(unique_words)}

        File.dump_json(metadata,meta_file_location)
        return meta_file_location





    @staticmethod
    def get_metadata(meta_file_location=None, df=None):
        meta_file_location = meta_file_location or data_prepo_meta
        if not os.path.isfile(meta_file_location):
            logger.debug("Meta data does not exists.")
            assert df is not None, "Cannot create meta with a None data frame, You should first create one using a " \
                                   "concrete data frame."
            VqaPredictor.create_meta(meta_file_location, df)
        meta_data = File.load_json(meta_file_location)
        # meta_data['ix_to_word'] = {str(word): int(i) for i, word in meta_data['ix_to_word'].items()}
        return meta_data


    @staticmethod
    def prepare_embeddings(embedding_filename=None, meta_file_location=None):
        embedding_filename = embedding_filename or embedding_matrix_filename
        metadata = VqaPredictor.get_metadata(meta_file_location)
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

    @staticmethod
    def get_embedding_matrix(embedding_filename=None, meta_file_location=None):
        embedding_filename = embedding_filename or embedding_matrix_filename
        if os.path.exists(embedding_filename ):
            logger.debug("Embedding Data already exists. Loading...")
            with h5py.File(embedding_filename ) as f:
                embedding_matrix = np.array(f['embedding_matrix'])
        else:
            embedding_matrix = VqaPredictor.prepare_embeddings(embedding_filename=embedding_filename,
                                                               meta_file_location=meta_file_location)
        return embedding_matrix
    @staticmethod
    def get_embedding_data(embedding_filename=None, meta_file_location=None):
        embedding_matrix = VqaPredictor.get_embedding_matrix(embedding_filename=embedding_filename, meta_file_location=meta_file_location)
        dim = embedding_dim
        s_length = seq_length
        meta_data = VqaPredictor.get_metadata(meta_file_location)

        return EmbeddingData(embedding_matrix=embedding_matrix,embedding_dim=dim, seq_length=s_length, meta_data=meta_data)

    def got_model(self, model):
        now = time.time()
        ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
        now_folder = os.path.abspath('{0}\\{1}\\'.format(vqa_models_folder, ts))
        model_fn = os.path.join(now_folder, 'vqa_model.h5')
        model_image_fn = os.path.join(now_folder, 'model_vqa.png5')
        summary_fn = os.path.join(now_folder, 'model_summary.txt')
        logger.debug("saving model to: '{0}'".format(model_fn))

        try:
            File.validate_dir_exists(now_folder)
            model.save(model_fn)  # creates a HDF5 file 'my_model.h5'
            logger.debug("model saved")


            print_model_summary_to_file(summary_fn, model)
            plot_model(model, to_file=model_image_fn)
        except Exception as ex:
            logger.error("Failed to save model:\n{0}".format(ex))