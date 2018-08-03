import tensorflow as tf

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
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
