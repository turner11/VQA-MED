import pandas as pd


class VqaSpecs(object):
    """"""

    @property
    def prediction_vector(self):
        df = pd.read_hdf(self.meta_data_location, self.prediction_df_name)
        assert len(df.columns) == 1, f'Expected prediction data frame to have a single column, but got: {df.columns}'
        vector = df[df.columns[0]]
        return vector

    def __init__(self, embedding_dim: int, seq_length: int, data_location: str, meta_data_location: str,
                 prediction_df_name: str = 'words') -> None:
        """"""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.data_location = data_location
        self.meta_data_location = meta_data_location
        self.prediction_df_name = prediction_df_name

    def __repr__(self):
        return f'{self.__class__.__name__}(embedding_dim={self.embedding_dim}, ' \
            f'seq_length={self.seq_length}, data_location={self.data_location}, ' \
            f'meta_data_location={self.meta_data_location}, prediction_df_name={self.prediction_df_name})'

    def __setstate__(self, state):
        self.__dict__.update(state)
