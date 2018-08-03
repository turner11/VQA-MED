from collections import namedtuple
from enum import Enum


class ClassifyStrategies(Enum):
    NLP = 1
    CATEGORIAL = 2

VqaSpecs = namedtuple('VqaSpecs',['embedding_dim', 'seq_length', 'data_location','meta_data'])