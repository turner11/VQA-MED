import itertools
from pathlib import Path
import pandas as pd

from parsers.data_loader import DataLoader


class DataLocations(object):
    """"""

    QA_BY_CATEGORY_FOLDER_NAME = 'QAPairsByCategory'

    def __init__(self, tag, base_folder):
        """"""
        super().__init__()
        self.tag = tag
        self._base_folder = base_folder
        self.images_folder = str(next(folder for folder in self.folder.iterdir() if 'images' in folder.name.lower()))

        category_path = Path(self.category_folder)
        if category_path.exists():
            categories_files = list(str(f) for f in category_path.iterdir())
        else:
            # For test set...
            categories_files = []

        self._all_qa_file_name = next(Path(self._base_folder).glob('All_QA_Pairs_*.txt|*Test_Questions.txt'), Path('FAILED'))

        base_path = Path(self._base_folder)
        _qa_file_names_patterns = ['All_QA_Pairs_*.txt','*Test_Questions.txt']
        candidates = itertools.chain.from_iterable(base_path.glob(pattern) for pattern in _qa_file_names_patterns)
        self._all_qa_file_name = next(candidates, Path('FAILED'))

        self.categories_files  = categories_files

        assert self.folder.exists()
        assert self._all_qa_file_name.exists()

    def __repr__(self):
        return f'{self.__class__.__name__}(base_folder={self._base_folder}, tag={self.tag})'

    @property
    def folder(self):
        return Path(self._base_folder)

    @property
    def qa_path(self):
        return str(self._all_qa_file_name)

    @property
    def category_folder(self):
        return str(self.folder / self.QA_BY_CATEGORY_FOLDER_NAME)

    def get_image_names_category_info(self):
        category_friendly_names = {'C1': 'Modality',
                                   'C2': 'Plane',
                                   'C3': 'Organ',
                                   'C4': 'Abnormality',}

        dfs = {}
        for fn in self.categories_files:
            category_sybol = Path(fn).name.split('_')[0]
            category = category_friendly_names[category_sybol]
            df = DataLoader.raw_input_to_dataframe(fn)

            df = df[['image_name', 'question']]
            df['question_category'] = category
            dfs[category] = df

        consolidated_category = pd.concat(list(dfs.values()))
        main_df = DataLoader.raw_input_to_dataframe(self._all_qa_file_name)
        main_df = pd.merge(left=main_df, right=consolidated_category
                           , left_on=['image_name', 'question'], right_on=['image_name', 'question']
                           , how='left')


        ret = main_df[['image_name', 'question','question_category']]
        return ret

        str()
