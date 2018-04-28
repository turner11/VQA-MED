class EmbeddingData(object):
    """"""

    @property
    def num_classes(self):
        from vqa_logger import logger
        val = 4
        logger.warning("using 4 just so I can keep going on. Fix this...")
        return val#len(self.meta_data['ix_to_ans'].keys())

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
        return super().__repr__()
