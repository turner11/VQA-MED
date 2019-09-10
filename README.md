# VQA-MED

This repo demonstrates the efforts made for the [ImageCLEF 2019 VQA-Med Q&A
](https://www.crowdai.org/challenges/imageclef-2019-vqa-med)  challenge.

As sepcified in the [ImageClef site](https://www.imageclef.org/2019/medical/vqa), the input for the model is constructed of an image + natural language question about said image, and an asnwer to the question.  
The task is to predict the answer for a similar data which the answer was ommitted from.  

You can see the the outline of the work done:  
1. (bringing data to expected format)[https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/0_bringing_data_to_expected_format.ipynb]  
2. [Pre process data](https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/1_pre_process_data.ipynb) (Clean + Enrich data)
3. [Data augmentation](https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/1.5_data_augmentation.ipynb)
4. [Create meta data](https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/2_create_meta_data.ipynb)
5. [Create the model](https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/3_creating_model.ipynb)
6. [Train the model](https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/4_training_model.ipynb)
7. [Predict](https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/5_predicting.ipynb)
8. [Create a submission in the expected format](https://github.com/turner11/VQA-MED/blob/master/VQA-MED/VQA.Python/6_create_submission.ipynb)

Please note, that this repo was created with the ImageClef contest in mind, and therfore was designed to work on my machine, and might need some tweeking to work on yours...

The resulting published paper is available [here](http://ceur-ws.org/Vol-2380/paper_116.pdf)


