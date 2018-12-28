
# coding: utf-8

# In[1]:


from vqa_logger import logger 
import IPython


# In[8]:


from classes.vqa_model_predictor import VqaModelPredictor, DefaultVqaModelPredictor
from common.DAL import get_models_data_frame, get_model
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.functions import get_highlited_function_code
df_models = get_models_data_frame()
df_show = df_models.sort_values(by=['wbss', 'bleu'], ascending=False).head()
df_show


# In[13]:


known_good_model = 85
model_id = known_good_model #df_show.id.iloc[0]
model_id = int(model_id)
mp = DefaultVqaModelPredictor(model_id)
mp


# In[14]:


mp.df_validation.head(2)


# In[15]:


code = get_highlited_function_code(mp.predict,remove_comments=False)
IPython.display.display(code)


# In[16]:


df_data = mp.df_validation
df_predictions = mp.predict(mp.df_validation)
df_predictions.head()


# In[8]:


df_predictions.describe()


# In[17]:


idx = 42
image_names = df_predictions.image_name.values
image_name = image_names[idx]

df_image = df_predictions[df_predictions.image_name == image_name]
# print(f'Result: {set(df_image.prediction)}')

image_path = df_image.path.values[0]
df_image


# In[19]:


from IPython.display import Image
Image(filename = image_path, width=400, height=400)


# In[20]:


df_image = df_data[df_data.image_name == image_name].copy().reset_index()
image_prediction = mp.predict(df_image)
image_prediction


# ## Evaluating the Model

# In[21]:


validation_prediction = df_predictions
predictions = validation_prediction.prediction.values
ground_truth = validation_prediction.answer.values
results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
print(f'Got results of\n{results}')


# In[22]:


validation_prediction.head(2)

