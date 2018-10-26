import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense, BatchNormalization, Activation
from keras import backend as K


def test_gpu():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # test tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    # test keras
    K.tensorflow_backend._get_available_gpus()


def compile_model():
    model_output_num_units = 2
    lstm_input_tensor = Input(shape=(10, 1), name='embedding_input')
    fc_tensors = Dense(units=16)(lstm_input_tensor)
    fc_tensors = BatchNormalization()(fc_tensors)
    fc_tensors = Activation('relu')(fc_tensors)

    fc_tensors = Dense(units=model_output_num_units, activation='sigmoid')(fc_tensors)

    fc_model = Model(inputs=[lstm_input_tensor], output=fc_tensors)

    fc_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return fc_model


def main():
    try:
        test_gpu()
        compile_model()
        print('Results: All OK!')
    except:
        print('Results: Failed!')
        raise


if __name__ == '__main__':
    main()
