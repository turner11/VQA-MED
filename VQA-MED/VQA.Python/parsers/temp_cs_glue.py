def get_models():
    from common import DAL

    models = DAL.get_models_data_frame()
    #'columns' # 'records'#'values'#'table'#'index'#'split'#
    j = models.to_json(orient='columns')

    return j

def main():
    get_models()

if __name__ == '__main__':
    main()

