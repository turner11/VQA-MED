#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from tqdm import tqdm
from classes.vqa_model_predictor import DefaultVqaModelPredictor
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.utils import VerboseTimer
import vqa_logger 
import logging
from pathlib import Path
import datetime
logger = logging.getLogger(__name__)


# In[2]:


# %%capture
# mp = DefaultVqaModelPredictor.get_contender()

main_model = 172
# specialized_classifiers = {'Abnormality': 172, 'Modality': 157, 'Organ': 160, 'Plane': 159, 'Abnormality_yes_no':178}

#with class weights (looks worse):
# specialized_classifiers = {'Abnormality': 172, 'Modality': 157, 'Organ': 180, 'Plane': 179, 'Abnormality_yes_no':178} 
# notes = 'Notes: with class weights (looks worse)'

#with class weights (looks worse):
specialized_classifiers = {'Abnormality': 172, 'Modality': 184, 'Organ': 183, 'Plane': 159, 'Abnormality_yes_no':178} 
specialized_classifiers = {'Abnormality': 186, 'Modality': 184, 'Organ': 183, 'Plane': 188, 'Abnormality_yes_no':178} 

main_model = 185
specialized_classifiers = {'Abnormality': 185, 'Modality': 184, 'Organ': 183, 'Plane': 188, 'Abnormality_yes_no':178} 
specialized_classifiers = {'Abnormality': 202, 'Modality': 184, 'Organ': 183, 'Plane': 188, 'Abnormality_yes_no':193} 
notes= 'Words prediction for abnormality - Optimized for BLEU'

mp = DefaultVqaModelPredictor(model=main_model, specialized_classifiers=specialized_classifiers)


# In[ ]:


mp


# In[ ]:


dd = mp.df_validation
category = 'Abnormality'
add = dd[dd.question_category==category].answer.drop_duplicates()
len(add)
# dd.head()


# In[ ]:


mp.model_folder.prediction_data_name


# In[ ]:


# %%capture
datasets = {'test':mp.df_test, 'validation':mp.df_validation}
df_name_to_predict = 'test'


predictions = {}

for name, df in datasets.items():
    with VerboseTimer(f"Predictions for VQA contender {name}"):
        df_predictions = mp.predict(df)
        predictions[name] = df_predictions



predictions['validation'][:5]


# In[ ]:


outputs = {}
for name, df_predictions in predictions.items():
    curr_predictions = df_predictions.prediction.values
    df_predicted = datasets[name]
    df_output = df_predicted.copy()
    df_output['image_id'] = df_output.path.apply(lambda p: p.rsplit(os.sep)[-1].rsplit('.', 1)[0])
    df_output['prediction'] = curr_predictions

    columns_to_remove = ['path',  'answer_embedding', 'question_embedding', 'group', 'diagnosis', 'processed_answer']
    for col in columns_to_remove:
        del df_output[col]

    sort_columns = sorted(df_output.columns, key=lambda c: c not in ['question', 'prediction', 'answer'])
    df_output = df_output[sort_columns]    
    outputs[name] = df_output


# In[ ]:


df_output_test = outputs['test']
df_output_validation = outputs['validation']


# In[ ]:


display = df_output_validation[df_output_validation.question_category == 'Abnormality']
display = df_output_validation
display.sample(10)


# In[ ]:


mp


# In[ ]:


len(df_output_test), len(df_output_test.image_id.drop_duplicates())


# In[ ]:


def get_str(df):
    strs = []
    debug_output_rows = df.apply(lambda row: row.image_id + '|'+ row.question + '|'+ row.prediction, axis=1 )
    output_rows = df.apply(lambda row: row.image_id + '|'+ row.prediction + '|'+row.answer, axis=1 )
    output_rows = output_rows.str.strip('|')
    rows = output_rows.values
    res = '\n'.join(rows)
    return res

res = get_str(df_output_test)
res_val = get_str(df_output_validation)


# In[ ]:


print(res[:200])


# ### Get evaluations per category:

# In[ ]:


evaluations = {}
pbar = tqdm(df_output_validation.groupby('question_category'))
for question_category, df in pbar:        
    pbar.set_description(f'evaluating {len(df)} for {question_category} items')
    curr_predictions = df.prediction.values
    curr_ground_truth = df.answer.values
    curr_evaluations = VqaMedEvaluatorBase.get_all_evaluation(predictions=curr_predictions, ground_truth=curr_ground_truth)
    evaluations[question_category] = curr_evaluations    

   


# ### Get Total Evaluation:

# In[ ]:


total_evaluations = VqaMedEvaluatorBase.get_all_evaluation(predictions=df_output_validation.prediction.values, ground_truth=df_output_validation.answer.values)    
evaluations['Total'] = total_evaluations


# In[ ]:


evaluations
df_evaluations = pd.DataFrame(evaluations).T#.sort_values(by=('bleu'))
df_evaluations['sort'] = df_evaluations.index == 'Total'
df_evaluations = df_evaluations.sort_values(by = ['sort', 'wbss'])
del df_evaluations['sort']
df_evaluations


# In[ ]:


model_repr = repr(mp)
model_repr
sub_models = {category: folder for category, (model, folder) in mp.model_by_question_category.items()}
sub_models_str = '\n'.join([str(f'{category}: {folder} ({folder.prediction_data_name})') for category, folder in sub_models.items() if folder is not None])

model_description_copy = df_evaluations.copy()

def get_prediction_vector(category):
    sub_model = sub_models.get(category)
    if sub_model is not None:
        return sub_model.prediction_data_name
    else:
        return '--'
    
model_description_copy['prediction_vector'] = model_description_copy.index.map(get_prediction_vector)


model_description =f'''
==Model==
{model_repr}

==Submodels==
{sub_models_str}

==validation evaluation==
{model_description_copy.to_string()}

==Notes==
{notes}
'''


print(model_description)


# In[ ]:


import time
now = time.time()
ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
submission_base_folder = Path('C:\\Users\\Public\\Documents\\Data\\2019\\submissions')
submission_folder = submission_base_folder/ts
submission_folder.mkdir()


txt_path = submission_folder/f'submission_{ts}.txt'
txt_path.write_text(res)

txt_path_val = submission_folder/f'submission_{ts}_validation.txt'
txt_path_val.write_text(res_val)


model_description_path = submission_folder/f'model_description.txt'
model_description_path.write_text(model_description)

with pd.HDFStore(str(submission_folder/ 'predictions.hdf')) as store:
    for name, df_predictions in predictions.items():
        store[name] = df_predictions


# In[ ]:


print(name)
df_predictions.sample(5)


# In[ ]:


idx = 14525
series = df_predictions.loc[idx]
series.answer, series.prediction ,series.probabilities 


# In[ ]:


mp

