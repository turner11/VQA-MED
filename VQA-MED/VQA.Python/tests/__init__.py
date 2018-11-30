import os

curr_folder, _ = os.path.split(__file__)
image_folder = os.path.join(curr_folder, 'test_images\\')
image_folder = os.path.normpath(image_folder)