import os
import pytest
import pandas as pd
import logging
from classes.vqa_model_builder import VqaModelBuilder
from classes.vqa_model_predictor import VqaModelPredictor
from classes.vqa_model_trainer import VqaModelTrainer
from common.utils import VerboseTimer
logger = logging.getLogger(__name__)


@pytest.fixture
def data_frame():
    folder, _ = os.path.split(__file__)
    data_path = os.path.abspath(os.path.join(folder,'data_for_test/train_data.hdf'))
    with pd.HDFStore(data_path) as store:
        train_data = store['data']
    return train_data

loss, activation = 'categorical_crossentropy', 'sigmoid'

def test_model_creation():
    with VerboseTimer("Instantiating VqaModelBuilder"):
        mb = VqaModelBuilder(loss, activation)

    with VerboseTimer("Instantiating VqaModelBuilder"):
        model = mb.get_vqa_model()

    model.summary()
    expected_fields = ['predict', 'fit']
    for field in expected_fields:
        assert hasattr(model, field), f'A model is expected to have a "{field}" attribute'


def test_model_training(data_frame):
    # Arrange
    train_data = data_frame
    train_data .group = 'train'
    train_data .group[::2] = 'validation'
    batch_size = 75

    mb = VqaModelBuilder(loss, activation)
    model = mb.get_vqa_model()
    mt = VqaModelTrainer(model, use_augmentation=False, batch_size=batch_size, data_location=train_data)

    # Act
    mt.train()

def test_model_predicting(data_frame):
    model_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\data_for_test\\test_model\\vqa_model_.h5'
    mp = VqaModelPredictor(model_path)
    test_data = data_frame
    predictions = mp.predict(test_data)
    preds = predictions.prediction
    # ground_truth = predictions.answer
    assert len(preds) == len(test_data), 'Got a different number for predictions input and output'


if __name__ == '__main__':
    # test_model_creation()
    df = data_frame()
    test_model_training(df)

