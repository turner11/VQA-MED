import os, sys
import tensorflow as tf

def test_gpu():
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


if __name__ == '__main__':
    from vqa_flow.main import main as main_flow
    from utils.describing import describe_models
    try:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        # test_gpu()
        # sys.exit(0)

        # models_folder = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\models'
        # describe_models(models_folder)
        # sys.exit(0)
        main_flow()
    except Exception as e:
        print("Got an error:\n{0}".format(e))
        raise
        # sys.exit(1)
