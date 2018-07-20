import spacy
from common.classes import ClassifyStrategies
from vqa_logger import logger

nlp=None

# How do we classify?-----------------------------------------------------------------------------
classify_strategy = ClassifyStrategies.CATEGORIAL
# classify_strategy = ClassifyStrategies.NLP


def get_stratagy_str():
    strategy_str = str(classify_strategy).split('.')[-1]
    return strategy_str


# NLP & Embedding-----------------------------------------------------------------------------
vectors = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']  # 'en_vectors_web_lg'
nlp_vector = vectors[2]

# glove_path =                    os.path.abspath('data/glove.6B.{0}d.txt'.format(embedding_dim))
# embedding_matrix_filename =     os.path.abspath('data/ckpts/embeddings_{0}.h5'.format(embedding_dim))
# ckpt_model_weights_filename = os.path.abspath('data/ckpts/model_weights.h5')
seq_length = 26
input_length = 32  # longest question / answer was 28 words. Rounding up to a nice round number\n
embedding_dim = 384  # Those are the sizes spacy uses
embedded_sentence_length = input_length * embedding_dim


def get_nlp():
    global nlp
    if nlp is None:
        logger.debug(f'using embedding vector: {nlp_vector }')
        nlp = spacy.load('en', vectors=nlp_vector)
        # logger.debug(f'vector "{nlp_vector}" loaded')
        # logger.debug(f'nlp creating pipe')
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # logger.debug(f'nlp getting embedding')
        # word_embeddings = nlp.vocab.vectors.data
        logger.debug(f'Got embedding')
    return nlp


# Image processing-----------------------------------------------------------------------------
DEFAULT_IMAGE_WIEGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}
image_size = image_size_by_base_models[DEFAULT_IMAGE_WIEGHTS]



import pandas as pd
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
