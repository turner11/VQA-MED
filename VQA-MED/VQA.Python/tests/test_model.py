import pytest
import os
import pandas as pd
import logging
from classes.vqa_model_builder import VqaModelBuilder
from classes.vqa_model_predictor import VqaModelPredictor
from classes.vqa_model_trainer import VqaModelTrainer
from common.utils import VerboseTimer
import tensorflow as tf

from data_access.model_folder import ModelFolder
from tests.conftest import model_path
import tests.conftest as conftest

tf.logging.set_verbosity(tf.logging.ERROR)

logger = logging.getLogger(__name__)


@pytest.fixture
def data_access():
    return conftest.data_access


@pytest.fixture
def data_frame():
    train_data = conftest.data_access.load_processed_data()
    return train_data

loss, activation = 'categorical_crossentropy', 'sigmoid'

@pytest.mark.filterwarnings('ignore:RuntimeWarning')
def test_model_creation():
    with VerboseTimer("Instantiating VqaModelBuilder"):
        mb = VqaModelBuilder(loss, activation)

    with VerboseTimer("Instantiating VqaModelBuilder"):
        model = mb.get_vqa_model()

    # model.summary()
    expected_fields = ['predict', 'fit']
    for field in expected_fields:
        assert hasattr(model, field), f'A model is expected to have a "{field}" attribute'


@pytest.mark.filterwarnings('ignore:RuntimeWarning')
def test_model_training(data_access):
    # Arrange
    batch_size = 5
    model_folder = ModelFolder(conftest.model_folder)

    mt = VqaModelTrainer(model_folder, augmentations=1, batch_size=batch_size, data_access=data_access)

    # Act
    mt.train()

# @pytest.mark.skip(reason='Still need to fix prediction for 2019 data')
@pytest.mark.filterwarnings('ignore:DeprecationWarning')
def test_model_predicting(data_frame):

    mp = VqaModelPredictor(model_path)
    test_data = data_frame
    predictions = mp.predict(test_data)
    preds = predictions.prediction
    # ground_truth = predictions.answer
    assert len(preds) == len(test_data), 'Got a different number for predictions input and output'


if __name__ == '__main__':
    # from common import settings
    conftest.pytest_runtest_setup(None)
    # from keras import backend as keras_backend
    # keras_backend.clear_session()
    # test_model_creation()
    df = data_frame()
    test_model_training(conftest.data_access)
    # test_model_training(df)
    test_model_predicting(df)

