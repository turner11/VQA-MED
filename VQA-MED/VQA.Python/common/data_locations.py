from pathlib import Path


class DataLocations(object):
    """"""

    QA_BY_CATEGORY_FOLDER_NAME = 'QAPairsByCategory'

    def __init__(self, tag, base_folder):
        """"""
        super().__init__()
        self.tag = tag
        self._base_folder = base_folder
        self.images_folder = str(next(folder for folder in self.folder.iterdir() if 'images' in folder.name))
        self.categories_files = list(str(f) for f in Path(self.category_folder).iterdir())

        self._all_qa_file_name = next(Path(self._base_folder).glob('All_QA_Pairs_*.txt'), Path('FAILED'))

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
