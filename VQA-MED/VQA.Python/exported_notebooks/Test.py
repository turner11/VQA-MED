import pandas as pd
# df = pd.DataFrame([['So', 'Long'],['Thanks','Fish'],['Traveler','Galaxy']], index=[3,2,1], columns=['X', 'Y'])

# from common.functions import get_image
# store_location =  'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5'
# k = 'augmentation_9'
# with pd.HDFStore(store_location) as store:
#     df_filtered = store[k].head()
# df_filtered['image'] = df_filtered.path.apply(lambda path: get_image(path))
# a = df_filtered['image'].values[0]
# str()

from common.os_utils import File

temp_index = File.load_pickle("D:\\Users\\avitu\\Downloads\\tempIndex.pkl")
df_index = File.load_pickle("D:\\Users\\avitu\\Downloads\\df.pkl")

t = temp_index
d = df_index
from classes.DataGenerator import DataGenerator
from common.constatns import data_location, vqa_specs_location

dg = DataGenerator(vqa_specs_location, shuffle=True)

le = len(dg)
X, y = dg[1367]
str()