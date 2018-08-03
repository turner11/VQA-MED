class EmbeddingData(object):
    """"""

    @property
    def num_classes(self):
        return len(self.meta_data['ix_to_ans'].keys())

    @property
    def num_words(self):
        return len(self.meta_data['ix_to_word'].keys())

    def __init__(self, embedding_matrix, embedding_dim, seq_length, meta_data):
        """"""
        super().__init__()
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.meta_data = meta_data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.embedding_matrix},{self.embedding_dim},{self.seq_length}, {self.meta_data})"
        # return super().__repr__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__:}(Embedding length:{len(self.embedding_matrix)}, Embedding dim: {self.embedding_dim}, seq length: {self.seq_length}, meta length: {len(self.meta_data)})"


