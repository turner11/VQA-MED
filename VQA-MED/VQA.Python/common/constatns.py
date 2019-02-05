import os
from pathlib import Path


# File Locations -------------------------------------------------------------

vqa_python_base_path = Path(os.path.join(os.path.abspath('.').split('VQA.Python')[0], 'VQA.Python'))
data_path = vqa_python_base_path / 'data'

images_folder_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images'
images_folder_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-images'

# The location to dump models to
vqa_models_folder = "C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models"

_DB_FILE_LOCATION = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\models_2019.db'

# -----------------------------------------------------------------------------
data_base_folder = Path('C:\\Users\\Public\\Documents\\Data\\2019')
data_path_train = str(data_base_folder / 'train')
data_path_validation = str(data_base_folder / 'validation')
data_path_test = str(data_base_folder / 'validation')

