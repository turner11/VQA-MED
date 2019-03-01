from classes.vqa_model_trainer import VqaModelTrainer
from common.settings import data_access
from common.utils import VerboseTimer
from data_access.model_folder import ModelFolder
import vqa_logger

import logging
logger = logging.getLogger(__name__)


# In[4]:

def do():
    model_location = 'C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190222_1346_47\\'
    model_folder = ModelFolder(model_location)

    batch_size = 75
    mt = VqaModelTrainer(model_folder, use_augmentation=True,batch_size=batch_size, data_access=data_access)
    history = mt.train()
    with VerboseTimer("Saving trained Model"):
        model_folder = mt.save(mt.model, mt.model_folder, history)
    print(model_folder.model_path)

