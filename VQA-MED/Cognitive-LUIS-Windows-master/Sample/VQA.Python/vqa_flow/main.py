# import keras
# import h5py
from data_access.data import DAL
from pre_processing.known_find_and_replace_items import all_tags, locations as locations_tags, diagnosis, \
    imaging_devices, locations
from vqa_flow.tags_classifier import TagClassifier
from vqa_flow.vqa_predictor import VqaPredictor
from vqa_logger import logger
from pre_processing.known_find_and_replace_items import train_data, validation_data

from vqa_flow.constatns import embedding_dim, seq_length


def main():
    # df = DAL.get_df(train_data)
    # embedding_data = VqaPredictor.get_embedding_data()
    # p = VqaPredictor()
    # vqa_model = p.get_vqa_model(embedding_data=embedding_data)
    #
    # return
    # model_path = "C:\\Users\\Public\\Documents\\Data\\2018\\models\\diagnosis_20180421_2056_09\\diagnosis_model.h5"
    # image_path_tumor ='C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images\\0392-100X-30-209-g002.jpg'
    # image_path_hematoma = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images\\13017_2015_52_Fig2_HTML.jpg'
    # TagClassifier.test_model(model_path, image_path_hematoma, diagnosis)
    #
    # return
    train_tags, do_images_to_embedded, words_2_embedded = [True, False, False]
    #### Train tags ----------------------------------------------------------------------------------------------------
    if train_tags:
        from utils.describing import main as d_main  # describe_models
        tag_sets = [('locations', locations), ('diagnosis',diagnosis), ('imaging_devices', imaging_devices)]

        for name, tags in tag_sets:
            logger.debug("Training model for: '{0}'".format(name))
            tc = TagClassifier(tags,name)
            tc.train_tags_model(train_data, validation_data)
            logger.debug("Done Training '{0}' model".format(name))


        # d_main()
    # #### END OF Train tags ----------------------------------------------------------------------------------------------------
    #
    # #### Images to embedded --------------------------------------------------------------------------------------------
    # if do_images_to_embedded:
    #     images_path_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images'
    #     images_path_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-images'
    #     images_path_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-images'
    #     # dump_path = images_to_embedded(images_path_test)
    # #### END OF Images to embedded--------------------------------------------------------------------------------------
    #
    # #### words to embedded ---------------------------------------------------------------------------------------------
    # if words_2_embedded:
    #     pass
    #     # path_to_weights = "C:\\Users\\Public\\Documents\\Data\\2017\\model_weights.h5"
    #     # with h5py.File(path_to_weights, "r") as f:
    #     #     embedding_matrix = f['/embedding_1/embedding_1_W']
    #     # #     for item in f.attrs.keys():
    #     # #         print("{0}: {1}\n".format(item ,f.attrs[item]))
    #     #     # str()
    #     #
    #     #     num_words = 12602#45185 #arbitraraly chose number of words in clef captions 2014 competition
    #     #     embedding_dim = 300#1000 # preaty much arbitrary
    #     #     seq_length = 5 # !!!!!! Not sure what is that...
    #     #     dropout_rate = 0.5 #picked up as naive default from an SO answer...
    #     #
    #     #     logger.debug("embedding_matrix shape: {0}".format(embedding_matrix.shape))
    #     #
    #     #     lstm_model = Word2VecModel(embedding_matrix=embedding_matrix, num_words=num_words, embedding_dim=embedding_dim,
    #     #                            seq_length=seq_length, dropout_rate=dropout_rate)
    #     #
    #     #
    #     #     results = try_run_lstm_model(lstm_model, seq_length)
    #     #     for k,v in results.items():
    #     #         print("{0}:\n\tag{1}\n\n".format(k,"\n".join([s for s in v])))
    # #### END OF words to embedded---------------------------------------------------------------------------------------
    #
    # # fileName = "prj_test.nexus.hdf5"
    # # f = h5py.File(fileName,  "r")
    # # for item in f.attrs.keys():
    # #     print(item + ":", f.attrs[item])
    # # mr = f['/entry/mr_scan/mr']
    # # i00 = f['/entry/mr_scan/I00']
    # # print("%s\tag%s\tag%s" % ("#", "mr", "I00"))
    # # for i in range(len(mr)):
    # #     print("%d\tag%g\tag%d" % (i, mr[i], i00[i]))
    # # f.close()


if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        print("Got an error:\n{0}".format(e))
        raise
        # sys.exit(1)



