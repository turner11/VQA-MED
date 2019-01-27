import os
from pathlib import Path

curr_folder, _ = os.path.split(__file__)
root = Path(curr_folder)
image_folder = str((root /'test_images\\').absolute())
model_path = str((root /'data_for_test\\test_model\\vqa_model_.h5').absolute())
