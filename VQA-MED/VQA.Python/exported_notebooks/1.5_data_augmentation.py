
# coding: utf-8

# In[1]:


import os
import pandas as pd
from pandas import HDFStore
import IPython
from IPython.display import Image, display
import pyarrow
from multiprocessing.pool import ThreadPool as Pool


# In[2]:


from common.constatns import data_location, vqa_specs_location, fn_meta, augmented_data_location
from common.utils import VerboseTimer
from common.functions import get_highlighted_function_code, generate_image_augmentations,  get_image
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


code = get_highlighted_function_code(generate_image_augmentations, remove_comments=False)
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
non_existing_paths = [(i, fn, file_name, ext, output_dir) for i, (fn, file_name, ext, output_dir) in enumerate(non_existing_paths)]


print(f'Generating augmentations for {len(non_existing_paths)} images')


def augments_single_image(tpl_data)  :
    try:       
        (i, curr_image_path, file_name, ext, output_dir) = tpl_data
        msg = (f'Augmenting ({i+1}/{len(non_existing_paths)})\t"{file_name}" -> {output_dir}')  
        if i %100 == 0:
            print(msg)
        File.validate_dir_exists(output_dir)
        generate_image_augmentations(curr_image_path, output_dir)
        res = 1
    except Exception as e: 
        msg = str(e)
        res = 0
    return (res,msg)


try:
    # for tpl_data in non_existing_paths:
         #augments_single_image(tpl_data)
    pool = Pool(processes=8)
    inputs = non_existing_paths
    pool_res = pool.map(augments_single_image, inputs)
    pool.terminate()

except Exception as ex:
    print(f'Error:\n{str(ex)}')


# In[13]:


failes = [tpl[1] for tpl in pool_res if tpl[0]==0]
successes = [tpl[1] for tpl in pool_res if tpl[0]==1]


f_summary = '\n'.join(failes[:5])
s_summary = '\n'.join(successes[:5])
summary = f'success: {len(successes)}\n{s_summary}\n\nfailes: {len(failes)}\n{f_summary}'.strip()

print(summary)


# In[14]:



a = images_info[:1]
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


# In[15]:


with VerboseTimer("Creating rows dataframe"):
    df_augmented_rows = pd.DataFrame(new_rows)
    
df = pd.concat([df_train, df_augmented_rows])    
print(len(df))

df.head(1)


# ## Giving a meaningful index across dataframes:

# In[16]:


df = df.sort_values(['augmentation', 'idx'], ascending=[True, True])


# In[17]:



len_df = len(df)
idxs = range(0, len_df)
len_idx = len(set(idxs))
assert  len_idx== len_df , f'length of indexes ({len_idx}) did not match length of dataframe ({len_df})'
df.idx = idxs


# In[18]:


df.iloc[[0,1,-2,-1]]


# In[19]:


data_location


# In[20]:


# # df.head(1)
# # len(new_rows)
# new_rows[1].augmentation
# df.columns
# aug_keys = df.augmentation.drop_duplicates().values

# aug_keys
df[['augmentation','idx']].iloc[[0,1,-2,-1]]


# In[21]:


import numpy as np
aug_keys = [int(i) if not np.isnan(i) else 0 for i in df.augmentation.drop_duplicates().values]
set(aug_keys)


# In[22]:


with HDFStore(data_location) as store:
       k = store.keys()
k        


# In[23]:



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

# In[24]:


with HDFStore(data_location) as store:
    loaded_index = store['index']

print(f'image_path: {loaded_index.image_path[0]}')    
print(f'store_path: {loaded_index.store_path[0]}')    
print(f'augmentation_key: {loaded_index.augmentation_key[0]}')    
  
loaded_index.head(1)


# In[25]:


with HDFStore(data_location) as store:
    print(list(store.keys()))


# In[26]:


with pd.HDFStore(data_location) as store:
    augmentation_1 = store['augmentation_1']
    augmentation_5 = store['augmentation_5']


# In[27]:


v5 = min(augmentation_5.idx),max(augmentation_5.idx)
v1 = min(augmentation_1.idx),max(augmentation_1.idx)

print(v5)
print(v1)
len(augmentation_1)
augmentation_1.head(5).idx


# In[28]:


augmentation_5.tail(5).idx

