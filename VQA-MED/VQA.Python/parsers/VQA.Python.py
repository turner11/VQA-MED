import json
from common.settings import data_access
from common.supress_print import SupressPrint

ERROR_KEY = "error"
result_columns = ['image_name', 'question', 'answer']
df_columns = ['image_name', 'question', 'answer', 'path', 'processed_question', 'processed_answer', 'diagnosis','question_category', 'group']
df_light = data_access._load_processed_data(columns=df_columns)

def service_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            with SupressPrint():
                df = func(*args, **kwargs)
            df_ret = df[result_columns]
            orientation='columns'#'records'#'values'#'table'#'index'#'split'#
            json_ret = df_ret.reset_index().to_json(orient=orientation)
        except Exception as ex:
            import traceback
            tb = traceback.format_exc()
            json_ret = json.dumps({ERROR_KEY: "Got an unexpected error:\n{0}\n\nTraceback:\n{1}".format(ex, tb)})
        finally:
            pass
        return json_ret
    return wrapper

@service_decorator
def get_image_data(image_name):
    df_ret = df_light[df_light.image_name == image_name]
    return df_ret


@service_decorator
def query_data(query_string):
    res_vec = [False] * len(df_light)
    for col in result_columns:
        res_vec = res_vec | df_light[col].str.contains(query_string, case=False)
    df_ret = df_light[res_vec]
    return df_ret



if __name__ == "__main__":
    # dd = get_image_data(image_name='synpic100295.jpg')
    # dd = get_image_data(image_name='synpic100295.jpg')
    # image = '0392-100X-31-222-g002.jpg'#'0392-100X-33-350-g002.jpg' #'0392-100X-31-109-g001.jpg'
    # ret_val = get_image_data(image_name=image)
    #
    # q = '0392-100X-31-222-g002.jpg'#'ct'
    # query_data(query_string=q)

    # print(ret_val)
    if False:
        parser = argparse.ArgumentParser(description='Extracts caption for a COCO image.')
        parser.add_argument('-p', dest='df_path', help='path of data frame',default=None)
        parser.add_argument('-n', dest='image_name', help='name_of_image')
        # parser.add_argument('-q', dest='query', help='query to look for',default=None)
        # parser.add_argument('-a', dest='question', help='question to ask',default=None)

        args = parser.parse_args()
        ret_val = get_image_data(image_name=args.image_name, dataframe_path=args.df_path)
        json_string = ret_val
        # json_string = json.dumps(ret_val)
        print(json_string)
        #raw_input()

