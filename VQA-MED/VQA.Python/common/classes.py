import warnings
from collections import namedtuple
from enum import Enum

from keras.callbacks import Callback


class ClassifyStrategies(Enum):
    NLP = 1
    CATEGORIAL = 2


VqaSpecs = namedtuple('VqaSpecs', ['embedding_dim', 'seq_length', 'data_location', 'meta_data_location'])


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current is None or self.value is None:
            pass
        elif current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True