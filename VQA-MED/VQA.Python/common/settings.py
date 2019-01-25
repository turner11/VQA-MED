from functools import lru_cache

import spacy
import logging

logger = logging.getLogger(__name__)

nlp = None
# Image processing-----------------------------------------------------------------------------
DEFAULT_IMAGE_WEIGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}
image_size = image_size_by_base_models[DEFAULT_IMAGE_WEIGHTS]

# NLP & Embedding-----------------------------------------------------------------------------
vectors = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']  # 'en_vectors_web_lg'

nlp_vector = vectors[0]


def set_nlp_vector(idx):
    global nlp_vector
    if nlp is not None:
        raise Exception('Changing the vector will have no affect once NLP was set')
    nlp_vector = vectors[idx]
    logger.debug(f'Embedding vector set to: {nlp_vector}')


seq_length = 26
input_length = 32  # longest question / answer was 28 words. Rounding up to a nice round number\n
embedding_dim = 384  # Those are the sizes spacy uses
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
