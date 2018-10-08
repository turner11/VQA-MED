import os
import random
import string
import datetime

import h5py
import numpy as np
import time

from keras import Model, models, Input, callbacks
from keras.utils import plot_model, to_categorical

from parsers.utils import VerboseTimer
from utils.os_utils import File, print_progress
from vqa_flow.constatns import embedding_matrix_filename, data_prepo_meta, embedding_dim, glove_path, seq_length, \
    vqa_models_folder
from vqa_flow.data_structures import EmbeddingData
from vqa_flow.image_models import ImageModelGenerator
from vqa_flow.keras_utils import print_model_summary_to_file
from vqa_logger import logger
from keras.layers import Dense, Embedding, LSTM, BatchNormalization#, GlobalAveragePooling2D, Merge, Flatten
import keras.layers as keras_layers
import itertools
from keras import backend as keras_backend
# from keras.models import Sequential #Model
# import pandas as pd


class VqaPredictor(object):



    def __init__(self, model_fn_path, training_data_object, validation_data_object):
        """"""
        super().__init__()
        self.__classes = None
        self.__model_fn_path = model_fn_path
        self.model = self._load_vqa_model(model_fn_path)
        self.training_data_object = training_data_object
        self.validation_data_object = validation_data_object

        self._meta = VqaPredictorFactory.get_metadata(self.training_data_object.meta_fn)
        self._meta_validation = VqaPredictorFactory.get_metadata(self.validation_data_object.meta_fn)



    def idx_to_class(self, idx):
        ix_to_ans = self._meta['ix_to_ans']
        class_ress = ix_to_ans[idx]
        return class_ress

    def _load_vqa_model(self, file_name):
        return models.load_model(file_name)

    def train(self, df):
        keras_backend.clear_session()
        model = self.model

        get_labels_idx = lambda tags: [self.classes.index(v) for v in tags]
        # convert the labels from integers to vectors
        categorial_train_labels = to_categorical(self.classes_indices, num_classes=self.class_count)
        categorial_validation_labels = to_categorical(get_labels_idx(validation_labels), num_classes=self.class_count)

        validation_data = (validation_features, categorial_validation_labels)

        ## construct the image generator for data augmentation
        # aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        #                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        #                                horizontal_flip=True, fill_mode="nearest")
        # train_generator = aug.flow(train_features, categorial_train_labels)

        stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,mode='auto')

        try:
            history = model.fit_generator(train_generator,
                                          validation_data=validation_data,
                                          steps_per_epoch=len(train_features) // self.batch_size,
                                          epochs=self.epochs,
                                          verbose=1,
                                          callbacks=[stop_callback],
                                          class_weight=class_weight
                                          )
            # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

            # history = model.fit(train_features,train_labels,
            #                     epochs=epochs,
            #                     batch_size=batch_size,
            #                     validation_data=validation_data)
        except Exception as ex:
            logger.error("Got an error training model: {0}".format(ex))
            model.summary(print_fn=logger.error)
            raise
        return model, history


class VqaPredictorFactory(object):
    """"""

    def __init__(self, embedding_matrix_path=None, image_model_initial_weights=None, merge_strategy='concat'):
        """"""
        assert callable(merge_strategy) or merge_strategy in ['mul','sum', 'concat', 'ave', 'dot', 'max']# 'cos',
        if isinstance(merge_strategy, str):
            strat_dict = {'mul': keras_layers.multiply,
                             'sum':keras_layers.add,
                            'concat':keras_layers.concatenate,
                            'ave': keras_layers.average,
                            # 'cos': keras_layers.co,
                            'dot': keras_layers.dot,
                            'max': keras_layers.maximum}
            strat_func = strat_dict[merge_strategy]
        else:
            strat_func = merge_strategy




        super(VqaPredictorFactory, self).__init__()
        self.image_model_initial_weights = image_model_initial_weights
        self.merge_strategy = strat_func
        self.embedding_matrix_path = embedding_matrix_path or embedding_matrix_filename

    def __repr__(self):
        return super(VqaPredictorFactory, self).__repr__()

    def word_2_vec_model(self, embedding_matrix, num_words, embedding_dim, seq_length, input_tensor):
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


    def get_vqa_model(self, embedding_data=None):
        embedding_data = embedding_data or VqaPredictorFactory.get_embedding_data()
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
            lstm_model = self.word_2_vec_model(embedding_matrix=embedding_matrix, num_words=num_words, embedding_dim=embedding_dim,
                                               seq_length=seq_length, input_tensor=lstm_input_tensor)

            logger.debug("Getting image model")
            out_put_dim = lstm_model.shape[-1].value
            image_input_tensor, image_model = ImageModelGenerator.get_image_model(self.image_model_initial_weights, out_put_dim=out_put_dim)


            logger.debug("merging final model")
            fc_tensors = self.merge_strategy(inputs=[image_model, lstm_model])
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
    #             image_infos = Image2Features(fn, base_model=base_model)
    #         except Exception as ex:
    #             image_infos = None
    #             logger.warning("Failed on embedded {0}:\n{1}".format(embeded_idx + 1, ex))
    #         return image_infos
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

    # def Image2Features(curr_image_path, base_model=None):
    #     model = get_image_model(base_model)
    #     x = image_to_preprocess_input(curr_image_path)
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
        metadata['ix_to_ans'] = {ans:i for ans, i in enumerate(set(df['answer']))}
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
            VqaPredictorFactory.create_meta(meta_file_location, df)
        meta_data = File.load_json(meta_file_location)
        # meta_data['ix_to_word'] = {str(word): int(i) for i, word in meta_data['ix_to_word'].items()}
        return meta_data


    @staticmethod
    def prepare_embeddings(embedding_filename=None, meta_file_location=None):
        embedding_filename = embedding_filename or embedding_matrix_filename
        metadata = VqaPredictorFactory.get_metadata(meta_file_location)
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
            embedding_matrix = VqaPredictorFactory.prepare_embeddings(embedding_filename=embedding_filename,
                                                                      meta_file_location=meta_file_location)
        return embedding_matrix
    @staticmethod
    def get_embedding_data(embedding_filename=None, meta_file_location=None):
        embedding_matrix = VqaPredictorFactory.get_embedding_matrix(embedding_filename=embedding_filename, meta_file_location=meta_file_location)
        dim = embedding_dim
        s_length = seq_length
        meta_data = VqaPredictorFactory.get_metadata(meta_file_location)

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
        except Exception as ex:
            logger.error("Failed to save model:\n{0}".format(ex))

        try:
            logger.debug("Writing history")
            print_model_summary_to_file(summary_fn, model)
            logger.debug("Done Writing History")
            logger.debug("Plotting model")
            plot_model(model, to_file=model_image_fn)
            logger.debug("Done Plotting")
        except Exception as ex:
            logger.warning("{0}".format(ex))