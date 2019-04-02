from functools import lru_cache
import spacy
import logging
from common.constatns import data_path_train, data_path_validation, data_path_test, data_path
from common.data_locations import DataLocations
from data_access.api import DataAccess

nlp = None
# data locations-----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
data_access = DataAccess(data_path)

train_data = DataLocations('train', base_folder=data_path_train)
validation_data = DataLocations('validation', base_folder=data_path_validation)
test_data = DataLocations('test', base_folder=data_path_test)

# Image processing-----------------------------------------------------------------------------
DEFAULT_IMAGE_WEIGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}
image_size = image_size_by_base_models[DEFAULT_IMAGE_WEIGHTS]

# NLP & Embedding-----------------------------------------------------------------------------
vectors = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']  # 'en_vectors_web_lg'

nlp_vector = vectors[0]


@lru_cache(1)
def set_nlp_vector(idx):
    global nlp_vector
    if nlp is not None:
        logger.warning('Changing the vector will have no affect once NLP was set')
    nlp_vector = vectors[idx]
    logger.debug(f'Embedding vector set to: {nlp_vector}')


# words length ((length, occurrences)): [(16, 76), (14, 13), (13, 154), (11, 2805), (10, 474), (9, 1754) ...]
# questions length ((length, occurrences)): [(4, 759),(5, 1762),(6, 2965), (7, 4623),(8, 2844), (9, 1416), (10, 380)...]
seq_length = 16
input_length = 12
embedding_dim = 384  # This is the sizes spacy uses
embedded_sentence_length = input_length * embedding_dim


@lru_cache(1)
def get_nlp():
    global nlp
    if nlp is None:
        logger.debug(f'using embedding vector: {nlp_vector}')
        try:
            nlp = spacy.load('en', vectors=nlp_vector)
        except OSError as ex:
            msg = f'Got an error while loading spacy vector: {ex}.\n' \
                f'Did you initialize it?\n' \
                f'Try running:\n' \
                f'"python -m spacy download en"'
            raise Exception(msg)
        # logger.debug(f'vector "{nlp_vector}" loaded')
        # logger.debug(f'nlp creating pipe')
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # logger.debug(f'nlp getting embedding')
        # word_embeddings = nlp.vocab.vectors.data
        logger.debug(f'Got NLP engine ({nlp_vector})')
    return nlp
