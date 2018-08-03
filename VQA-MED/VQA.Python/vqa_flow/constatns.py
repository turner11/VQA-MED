import os
seq_length =    26
embedding_dim = 300

glove_path =                    os.path.abspath('data/glove.6B.{0}d.txt'.format(embedding_dim))
embedding_matrix_filename =     os.path.abspath('data/ckpts/embeddings_{0}.h5'.format(embedding_dim))
ckpt_model_weights_filename =   os.path.abspath('data/ckpts/model_weights.h5')

# Fail fast...
suffix = "Failing fast:\n"
assert os.path.isfile(glove_path), suffix+"glove file does not exists:\n{0}".format(glove_path)
# assert os.path.isfile(embedding_matrix_filename), suffix+"Embedding matrix file does not exist:\n{0}".format(embedding_matrix_filename)
assert os.path.isfile(ckpt_model_weights_filename), suffix+"glove file does not exists:\n{0}".format(ckpt_model_weights_filename)

data_prepo_meta = os.path.abspath('data/my_data_prepro.json')
data_prepo_meta_validation = os.path.abspath('data/my_data_prepro_validation.json')
data_prepo_meta_REFERENCE =           os.path.abspath('data/data_prepro.json')
# model_weights_filename =    os.path.abspath('data/model_weights.h5')
#
# data_img =                  os.path.abspath('data/data_img.h5')
# data_prepo =                os.path.abspath('data/data_prepro.h5')

vqa_models_folder = "C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models"