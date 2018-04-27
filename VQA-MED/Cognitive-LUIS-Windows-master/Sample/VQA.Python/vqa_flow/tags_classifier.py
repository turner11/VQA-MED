# import os
# import math
import numpy as np
import os
import time
import datetime

import imutils
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from data_access.data import DAL
from parsers.VQA18 import Vqa18Base
from pre_processing.known_find_and_replace_items import models_folder
from utils.os_utils import File
from sklearn.utils import class_weight as sk_learn_class_weight
from keras import callbacks, optimizers, backend as keras_backend, models
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Sequential#, Model
from keras.layers import Dense, Dropout, Flatten  # , Embedding, LSTM, Merge

from vqa_flow.image_models import ImageModelGenerator
from vqa_logger import logger


class TagClassifier(object):
    IMAGE_SIZE = (224, 224)  # (28,28)# IMAGE_NET expect this size

    @property
    def class_count(self):
        return len(self.classes)

    def __init__(self, classes_to_clasify, name, batch_size=20, epochs=25, image_model_initial_weights=None):
        """"""
        self.metrics = "accuracy"
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = classes_to_clasify
        self.image_model_initial_weights = image_model_initial_weights

    @classmethod
    def test_model(cls, model_arg, image_path, labels):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        img = cv2.imread(image_path)
        orig = img.copy()

        img = cls.image_to_pre_process_input(image_path)

        img = np.expand_dims(img, axis=0)
        # load the trained convolutional neural network
        logger.info("loading network...")
        if isinstance(model_arg, str) and os.path.isfile(model_arg):
            model = models.load_model(model_arg)
        else:
            model = model_arg

        # classify the input img
        encoded_classes = model.predict(img)[0]
        # build the label
        label_idx = np.argmax(encoded_classes)
        label = labels[label_idx]
        proba = encoded_classes[label_idx]
        label = "{}: {:.2f}%".format(label, proba * 100)

        # draw the label on the img
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # show the output img
        cv2.imshow("Output", output)
        cv2.waitKey(0)

    @classmethod
    def image_to_pre_process_input(cls, image_path):
        img = image.load_img(image_path, target_size=cls.IMAGE_SIZE)
        x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # x = np.array(data, dtype="float") / 255.0
        return x

    def train_tags_model(self, train_arg, validation_arg):
        # from pre_processing.known_find_and_replace_items import train_data, validation_data, test_data
        def get_nn_inputs(data_arg):
            df = DAL.get_df(data_arg)
            df_tags = df[self.classes]

            # Use only rows that has exactly 1 tag
            df_single_tag = df_tags[df_tags.apply(sum, axis=1) == 1]
            decoded_labels = df_single_tag.idxmax(axis=1)
            labels = np.asanyarray(decoded_labels)

            relevant_images = df.loc[decoded_labels.index][Vqa18Base.COL_IMAGE_NAME]
            images_files = [os.path.join(data_arg.images_path, fn) + ".jpg" for fn in relevant_images]
            images_files = [fn for fn in images_files if os.path.isfile(fn)]

            # load the image, pre-process it, and store it in the data list
            features_list = [self.image_to_pre_process_input(fn) for fn in images_files]
            features = np.asanyarray(features_list)

            assert len(labels) == len(features), \
                "Got different number of labels ({0}) and features ({1})".format(len(labels), len(features))
            return features, labels

        validation_features, validation_labels = get_nn_inputs(validation_arg)
        train_features, train_labels = get_nn_inputs(train_arg)

        model, history = self.train_nn(train_features=train_features
                                       , train_labels=train_labels
                                       , validation_features=validation_features
                                       , validation_labels=validation_labels)  # .tolist()

        self.save_tags_model(model, history)
        return model, history


    def train_nn(self, train_features, train_labels, validation_features, validation_labels):
        # Making sure to release memory from previous session, in case they did not end properly....
        keras_backend.clear_session()

        model = self.__get_model()

        get_labels_idx = lambda tags: [self.classes.index(v) for v in tags]
        # convert the labels from integers to vectors
        categorial_train_labels = to_categorical(get_labels_idx(train_labels), num_classes=self.class_count)
        categorial_validation_labels = to_categorical(get_labels_idx(validation_labels), num_classes=self.class_count)

        validation_data = (validation_features, categorial_validation_labels)

        # construct the image generator for data augmentation
        aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                       height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode="nearest")
        train_generator = aug.flow(train_features, categorial_train_labels)

        early_stop_call_back = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,
                                                       mode='auto')

        # Just for understading data...

        y_ints = [y.argmax() for y in categorial_train_labels]
        class_weight = sk_learn_class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
        labels_dict = {i: int(sm) for i, sm in enumerate(categorial_train_labels.sum(axis=0))}
        logger.debug("Labels Count was:\n{0}\nClasses Weights was:\n{1}".format(labels_dict, class_weight))

        try:
            history = model.fit_generator(train_generator,
                                          validation_data=validation_data,
                                          steps_per_epoch=len(train_features) // self.batch_size,
                                          epochs=self.epochs,
                                          verbose=1,
                                          callbacks=[early_stop_call_back],
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

    def __get_model(self):
        DENSE_UNITS = 1024
        DENSE_ACTIVATION = 'relu'
        dropout_rate = 0.5
        height, width = self.IMAGE_SIZE
        depth = 3  #

        #input_shape = (height, width, depth)

        ## if we are using "channels first", update the input shape
        #if keras_backend.image_data_format() == "channels_first":
            #input_shape = (depth, height, width)

        ## model.add(layers.Conv2D(20, (5, 5), padding="same", input_shape=input_shape, name="Conv2D_1"))

        # initialize the model
        model = Sequential()
        image_model = ImageModelGenerator.get_image_model(self.image_model_initial_weights)
        model.add(image_model)
        # -----------------------------------------------------
        model.add(Dropout(rate=dropout_rate, name="dropout_1"))
        model.add(Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION, name='dense_2'))
        model.add(Dropout(rate=dropout_rate, name="dropout_3"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(units=self.class_count, activation='softmax', name='output_dense_4'))
        # -----------------------------------------------------
        INIT_LR = 1e-3
        decay = INIT_LR / self.epochs
        opt = optimizers.Adam(lr=INIT_LR, decay=decay)
        # opt = optimizers.RMSprop(lr=2e-4)
        loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer=opt, metrics=[self.metrics])

        # return the constructed network architecture

        return model

    def save_tags_model(self, model, history=None):
        now = time.time()
        ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
        now_folder = os.path.abspath('{0}\\{1}_{2}\\'.format(models_folder, self.name, ts))
        model_fn = os.path.join(now_folder, '{0}_model.h5'.format(self.name))
        summary_fn = os.path.join(now_folder, 'model_summary.txt')
        logger.debug("saving model to: '{0}'".format(model_fn))

        try:
            File.validate_dir_exists(now_folder)
            model.save(model_fn)  # creates a HDF5 file 'my_model.h5'
            logger.debug("model saved")
            File.write_text(summary_fn, model.summary(print_fn=logger.debug()))
        except Exception as ex:
            logger.error("Failed to save model:\n{0}".format(ex))

        if history:
            plot_fn = os.path.join(now_folder, 'model_plot.png')
            logger.debug("saving model plot to: '{0}'".format(plot_fn))
            try:
                plt.style.use("ggplot")
                plt.figure()

                loss = history.history["loss"]
                val_loss = history.history["val_loss"]
                acc = history.history["acc"]
                val_acc = history.history["val_acc"]

                N = len(history.history["val_acc"])
                epochs = np.arange(1, N + 1)

                plt.plot(epochs, loss, label="train_loss")
                plt.plot(epochs, val_loss, label="val_loss")
                plt.plot(epochs, acc, label="train_acc")
                plt.plot(epochs, val_acc, label="val_acc")

                validation_final_accuracy = val_acc[-1]
                plt.title("Training Loss and Accuracy on '{0}'\nValidation Accuracy: {1}".format(self.name,
                                                                                                 validation_final_accuracy))

                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend(loc="lower left")
                plt.savefig(plot_fn)
                logger.debug("plot saved")

                history_fn = os.path.join(now_folder, 'history.pkl')
                File.dump_pickle(history, history_fn)
            except Exception as ex:
                logger.warning("Failed to save plot:\n{0}".format(ex))

            try:
                history_fn = os.path.join(now_folder, 'history.pkl')
                logger.debug("saving history to: '{0}'".format(history_fn))
                File.dump_pickle(history, history_fn)

            except Exception as ex:
                logger.warning("Failed to save history:\n{0}".format(ex))



