from data_access.api import DataAccess
import os
from pathlib import Path

curr_folder, _ = os.path.split(__file__)
root = Path(curr_folder)
image_folder = str((root /'test_images\\').absolute())
data_folder = root /'data_for_test'
model_folder = data_folder / 'test_model'

model_path = str((data_folder / 'test_model\\vqa_model.h5').absolute())

data_access: DataAccess = None

def pytest_runtest_setup(item):
    # __generate_data_folder()
    global data_access
    if data_access is None:
        da = DataAccess(data_folder)
        data_access = da


def __generate_data_folder():
    # Just for having test data
    from common.settings import data_access as dd
    da = DataAccess(data_folder)
    import pandas as pd
    dr = dd.load_processed_data()
    groups = dr['group'].drop_duplicates().values
    dfs = []
    for group in groups:
        df = dr[dr['group'] == group]
        question_categorys = df.question_category.drop_duplicates().values
        for question_category in question_categorys:
            df_2 = df[df.question_category == question_category].head(5)
            dfs.append(df_2)
    res = pd.concat(dfs)
    print(len(res))
    da.save_processed_data(res)
    return da


def main():
    pytest_runtest_setup(None)

    from tests.test_model import test_model_training
    test_model_training(data_access)



if __name__ == '__main__':
    main()
