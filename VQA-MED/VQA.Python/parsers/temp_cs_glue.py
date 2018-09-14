import json
from common.supress_print import supress_print

model = None
def get_models():
    from common import DAL

    models = DAL.get_models_data_frame()
    #'columns' # 'records'#'values'#'table'#'index'#'split'#
    j = models.to_json(orient='columns')

    return j


@supress_print
def set_model(model_id):
    global model


    try:

        from classes.vqa_model_predictor import VqaModelPredictor
        mp = VqaModelPredictor(model_id)
        model = mp
        success = True
    except:
        success = False
    return success




    # def predict(question, image_name):
#     # model.df_test
#     df_predictions = model.df_test#model.df_validation
#     df_image = df_predictions[df_predictions.image_name == image_name]
#     # print(f'Result: {set(df_image.prediction.values)}')
#     return json.dumps(df_image)



def main():
    set_model(34)
    # get_models()

if __name__ == '__main__':
    main()

