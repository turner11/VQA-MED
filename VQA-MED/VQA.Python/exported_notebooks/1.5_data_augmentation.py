#!/usr/bin/env python
# coding: utf-8

# # Data Augmentation

# In this notebook will will generate more data to train on based on given train and validation sets.  

# ### Some main functions we used:

# In[1]:


import IPython
from common.functions import get_highlighted_function_code


# #### The augmentation function:

# In[2]:


from common.functions import generate_image_augmentations
code = get_highlighted_function_code(generate_image_augmentations,remove_comments=False)
IPython.display.display(code)  


# ---
# ## The code:

# In[3]:


import os
import pandas as pd
import IPython
from IPython.display import Image, display
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
import logging
from collections import defaultdict, namedtuple
from pathlib import Path


# In[4]:


from common.utils import VerboseTimer
from common.functions import get_highlighted_function_code, generate_image_augmentations,  get_image
from common.os_utils import File
from common.settings import data_access
import vqa_logger 
logger = logging.getLogger(__name__)


# In[5]:


df_data = data_access.load_processed_data(columns=['path','question','answer', 'group'])


# In[6]:


df_data = df_data[df_data.group.isin(['train','validation'])]
print(f'Data length: {len(df_data)}')        
df_data.head(2)


# In[7]:


df_data.group.drop_duplicates()


# ### For the augmaentation we will use the following code:

# In[8]:


df_train = df_data[df_data.group == 'train']

image_paths = df_train.path.drop_duplicates()
print(len(image_paths))

ImageInfo = namedtuple('ImageInfo',
                       ['original_path', 'file_name', 'extension', 'target_location', 'out_put_folder_exists'])


def get_file_info(fn):
    image_folder, full_file_name = os.path.split(fn)
    file_name, ext = full_file_name.split('.')[-2:]
    output_dir = os.path.join(image_folder, 'augmentations', full_file_name + '\\')
    output_exists = os.path.isdir(output_dir)
    return ImageInfo(fn, file_name, ext, output_dir, output_exists)


images_info = [get_file_info(p) for p in image_paths]
df_all_images_info = pd.DataFrame(images_info)
df_images_info = df_all_images_info[~df_all_images_info.out_put_folder_exists]

print(f'Generating augmentations for {len(df_images_info)} images')


def augments_single_image(row_index):
    try:
        row = df_images_info.iloc[row_index]
        msg = (f'Augmenting ({row_index + 1}/{len(df_images_info)})\t"{row.file_name}" -> {row.target_location}')
        if row_index % 100 == 0:
            print(msg)
        File.validate_dir_exists(row.target_location)
        generate_image_augmentations(row.original_path, row.target_location)
        res = 1
    except Exception as e:
        msg = str(e)
        res = 0
    return (res, msg)


# for tpl_data in non_existing_paths:
# augments_single_image(tpl_data)
pool = Pool(processes=8)
inputs = range(len(df_images_info))
pool_res = pool.map(augments_single_image, inputs)
pool.terminate()


# In[10]:


failes = [tpl[1] for tpl in pool_res if tpl[0]==0]
successes = [tpl[1] for tpl in pool_res if tpl[0]==1]


f_summary = '\n'.join(failes[:5])
s_summary = '\n'.join(successes[:5])
summary = f'success: {len(successes)}\n{s_summary}\nfailes: {len(failes)}\n{f_summary}'.strip()

print(summary)


# In[11]:


df_all_images_info.head()
# len(df_all_images_info.original_path.drop_duplicates()), len(df_all_images_info), len(df_all_images_info.drop_duplicates())


# In[12]:



# Set the original path
df_augments = df_train[['path']].drop_duplicates().copy()
df_augments['augmentation'] = 0
df_augments['original_path'] = df_augments.path

print(len(df_augments))

# Add the augmentations
new_rows = []
AugmentationRow = namedtuple('AugmentationRow',['original_path', 'path', 'augmentation'])
index = df_all_images_info[['original_path','target_location']].set_index('original_path')
with VerboseTimer("Collecting augmented rows"):
    pbar = tqdm(df_augments.iterrows(), total=len(df_augments))
    for i, row in pbar:
        augment_location = Path(index.loc[row.original_path].target_location)
        assert augment_location.exists()
        augment_files = sorted(augment_location.iterdir())

        curr_augmentations = [AugmentationRow(row.original_path, path=str(augmented_file),augmentation=i)
                              for i, augmented_file
                              in enumerate(augment_files, start=1)] # 0 is for the original
        new_rows.extend(curr_augmentations)


# Last preperatons (sorting, data types...)

# In[15]:


df = df_augments.append(new_rows)
df['augmentation'] = df.augmentation.astype(int)
df = df.sort_values(['augmentation'], ascending=[True])
# print(len(df), len(df.drop_duplicates()))
assert len(df) ==  len(df.drop_duplicates()), 'got duplicated row'


# And lets take a look:

# In[17]:


df.iloc[[0,1,-2,-1]]


# #### Saving the data

# In[29]:


data_access.save_augmentation_data(df)


# ### The results:

# In[21]:


augmentation_1 = data_access.load_augmentation_data(augmentations=1)
augmentation_5 = data_access.load_augmentation_data(augmentations=5)
augmentation_all = data_access.load_augmentation_data()
print(len(augmentation_all))
augmentation_all.sample(5)


# Validation of data:

# In[22]:


orig_a1 = set(augmentation_1.original_path)
orig_a5 = set(augmentation_5.original_path)

diff = orig_a1 ^ orig_a5
diff
print(len(orig_a1))
assert len(diff) == 0, 'Expected all augmentations to have all orignal paths'

