import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense, BatchNormalization, Activation
from tensorflow.python.client import device_lib
from keras import backend as K


def test_gpu():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # test tf
    string_arg = 'Hello, TensorFlow!'
    hello = tf.constant(string_arg )
    sess = tf.Session()
    hello_string = sess.run(hello)
    hello_string = str(hello_string, 'utf-8')

    assert hello_string  == string_arg , 'Got wring string'

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # local_devices = device_lib.list_local_devices()
    # str_local_devices = str(local_devices)
    # has_gpu = str_local_devices.lower().find('/device:gpu') >=0

    avaliable_gpus = K.tensorflow_backend._get_available_gpus()
    has_gpu = len(avaliable_gpus) > 0
    assert has_gpu, "No GPUs were found"


def test_compile_model():
    # Just check not raising Exceptions
    model_output_num_units = 2
    lstm_input_tensor = Input(shape=(10, 1), name='embedding_input')
    fc_tensors = Dense(units=16)(lstm_input_tensor)
    fc_tensors = BatchNormalization()(fc_tensors)
    fc_tensors = Activation('relu')(fc_tensors)
    fc_tensors = Dense(units=model_output_num_units, activation='sigmoid')(fc_tensors)
    fc_model = Model(inputs=[lstm_input_tensor], output=fc_tensors)
    fc_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])



def main():
    try:
        test_gpu()
        test_compile_model()
        print('Results: All OK!')
    except:
        print('Results: Failed!')
        raise


if __name__ == '__main__':
    main()
