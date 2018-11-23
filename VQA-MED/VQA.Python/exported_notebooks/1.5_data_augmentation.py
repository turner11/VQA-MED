
# coding: utf-8

# In[1]:


import os
import pandas as pd
from pandas import HDFStore
import IPython
from IPython.display import Image, display
import pyarrow


# In[2]:


from common.constatns import data_location, vqa_specs_location, fn_meta, augmented_data_location
from common.utils import VerboseTimer
from common.functions import get_highlited_function_code, generate_image_augmentations,  get_image
from common.os_utils import File


# In[3]:


print(f'loading from:\n{data_location}')
with VerboseTimer("Loading Data"):
    with HDFStore(data_location) as store:
         df_data = store['data']
        
df_data = df_data[df_data.group.isin(['train','validation'])]
print(f'Data length: {len(df_data)}')        
df_data.head(2)


# ### For the augmaentation we will use the following code:

# In[4]:


code = get_highlited_function_code(generate_image_augmentations,remove_comments=False)
IPython.display.display(code)  


# In[5]:


df_train = df_data[df_data.group == 'train']

image_paths = df_train.path.drop_duplicates()
print(len(image_paths))




def get_file_info(fn):
        image_folder, full_file_name = os.path.split(fn)
        file_name, ext = full_file_name.split('.')[-2:]        
        output_dir = os.path.join(image_folder,'augmentations',full_file_name+'\\')
        return (fn, file_name, ext, output_dir)
        
images_info = [get_file_info(p) for p in image_paths]        
non_existing_paths = [(fn, file_name, ext, output_dir) for (fn, file_name, ext, output_dir) in images_info if not os.path.isdir(output_dir)]



print(f'Generating augmentations for {len(non_existing_paths)} images')

non_existing_paths = non_existing_paths
for i,  (curr_image_path, file_name, ext, output_dir) in enumerate(non_existing_paths):
    print(f'Augmenting ({i+1}/{len(non_existing_paths)})\t"{file_name}" -> {output_dir}')    
    File.validate_dir_exists(output_dir)
    generate_image_augmentations(curr_image_path, output_dir)


# In[6]:


aa = images_info[:1]
a = images_info
aug_dict = {image_path:output_dir for (image_path, file_name, ext, output_dir) in a}

curr_idx = df_train.tail(1).index[0] +1

df_augments = df_train.copy()
df_augments['augmentation'] = 0
df_augments['idx'] = 0

print(len(df_augments))
new_rows = []
with VerboseTimer("Collecting augmented rows"):
    for image_path, output_dir in aug_dict.items():
        #print(image_path)
        image_rows = df_augments[df_augments.path == image_path]
        for i_row, row in image_rows.iterrows():
            #print(i_row)
            augment_files = [os.path.join(output_dir, fn) for fn in sorted(os.listdir(output_dir))]

            for i_augment, augment_path in enumerate(augment_files):
                r = row.copy()
                r.path = augment_path            
#                 r.image = get_image(augment_path)
                r.augmentation = i_augment + 1 
                r.idx = curr_idx
                curr_idx+=1
                r.reset_index()
                new_rows.append(r)        


# In[7]:


with VerboseTimer("Creating rows dataframe"):
    df_augmented_rows = pd.DataFrame(new_rows)
    
df = pd.concat([df_train, df_augmented_rows])    
print(len(df))

df.head(0)


# ## Giving a meaningful index across dataframes:

# In[8]:


df = df.sort_values(['augmentation', 'idx'], ascending=[True, True])


# In[9]:



len_df = len(df)
idxs = range(0, len_df)
len_idx = len(set(idxs))
assert  len_idx== len_df , f'length of indexes ({len_idx}) did not match length of dataframe ({len_df})'
df.idx = idxs


# In[10]:


df.iloc[[0,1,-2,-1]]


# In[11]:


data_location


# In[12]:


# # df.head(1)
# # len(new_rows)
# new_rows[1].augmentation
# df.columns
# aug_keys = df.augmentation.drop_duplicates().values

# aug_keys
df[['augmentation','idx']].iloc[[0,1,-2,-1]]


# In[13]:


import numpy as np
aug_keys = [int(i) if not np.isnan(i) else 0 for i in df.augmentation.drop_duplicates().values]
set(aug_keys)


# In[14]:


with HDFStore(data_location) as store:
       k = store.keys()
k        


# In[15]:



from collections import defaultdict
index_dict = defaultdict(lambda:[])

with VerboseTimer(f"Storing {len(aug_keys)} dataframes"):
    with HDFStore(data_location) as store:
        for aug_key in aug_keys:
            with VerboseTimer(f"Storing dataframe '{aug_key}'"):
                data = df[df.augmentation == aug_key]

                store_key = f'augmentation_{aug_key}'
                idxs = data.idx.values                                
                index_dict['idx'].extend(idxs)        
                
                paths = data.path.values                
                index_dict['paths'].extend(paths)                
                
                index_dict['image_path'].extend(paths)
                index_dict['augmentation_key'].extend([aug_key]*len(paths))
                index_dict['store_path'].extend([data_location]*len(paths))
                index_dict['store_key'].extend([store_key]*len(paths))
                store[store_key] = data
                
        index=pd.DataFrame(index_dict) 
        store['index'] = index


# ### The results:

# In[16]:


with HDFStore(data_location) as store:
    loaded_index = store['index']

print(f'image_path: {loaded_index.image_path[0]}')    
print(f'store_path: {loaded_index.store_path[0]}')    
print(f'augmentation_key: {loaded_index.augmentation_key[0]}')    
  
loaded_index.head(1)


# In[17]:


with HDFStore(data_location) as store:
    print(list(store.keys()))


# In[18]:


with pd.HDFStore(data_location) as store:
    augmentation_1 = store['augmentation_1']
    augmentation_20 = store['augmentation_20']


# In[19]:


v20 = min(augmentation_20.idx),max(augmentation_20.idx)
v1 = min(augmentation_1.idx),max(augmentation_1.idx)

print(v20)
print(v1)
len(augmentation_1)
augmentation_1.head(5).idx


# In[21]:


augmentation_20.tail(5).idx

