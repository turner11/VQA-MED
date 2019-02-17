import os
from pathlib import Path
import vqa_logger

# File Locations -------------------------------------------------------------
vqa_python_base_path = Path(os.path.join(os.path.abspath('.').split('VQA.Python')[0], 'VQA.Python'))
data_path = vqa_python_base_path / 'data'
# -----------------------------------------------------------------------------
base_data_folder = Path('C:\\Users\\Public\\Documents\\Data\\2019')
data_path_train = str(base_data_folder / 'train')
data_path_validation = str(base_data_folder / 'validation')
data_path_test = str(base_data_folder / 'validation')

# The location to dump models to
vqa_models_folder = str(base_data_folder / 'models')
# vqa_models_folder = "C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models"
_DB_FILE_LOCATION = str(vqa_python_base_path / 'models_2019.db')  # The root of python project...
