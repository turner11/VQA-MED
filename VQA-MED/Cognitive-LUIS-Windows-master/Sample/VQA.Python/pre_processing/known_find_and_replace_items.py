from collections import namedtuple
import itertools

FindAndReplaceData = namedtuple('FindAndReplaceData',['orig','sub'])

find_and_replace_collection = [FindAndReplaceData('magnetic resonance imaging', 'MRI'),
                               FindAndReplaceData('magnetic resonance angiography', 'MRA'),
                               FindAndReplaceData('mri', 'MRI'),
                               FindAndReplaceData('ct', 'CT'),
                               FindAndReplaceData(' mra ', ' MRA '),
                               FindAndReplaceData(' ct scan ', ' CT '),
                               FindAndReplaceData(' mri scan ', ' MRI '),
                               FindAndReplaceData(' ct scan', ' CT '),
                               FindAndReplaceData(' mri scan', ' MRI '),
                               FindAndReplaceData(' ct image ', ' CT '),
                               FindAndReplaceData(' mri image ', ' MRI '),
                               FindAndReplaceData(' ct image', ' CT '),
                               FindAndReplaceData(' mri image', ' MRI '),
                               FindAndReplaceData('the CT', 'CT'),
                               FindAndReplaceData('the MRI', 'MRI'),
                               FindAndReplaceData('the', ''),
                               FindAndReplaceData('and', ''),
                               FindAndReplaceData('in', ''),
                               FindAndReplaceData('of', ''),
                               FindAndReplaceData('reveal', 'show'),
                               FindAndReplaceData('reveals', 'show'),
                               FindAndReplaceData('lesion', 'tumor'),
                               FindAndReplaceData('Cerebral', 'brain'),
                               FindAndReplaceData('  ', ' '),
                               #



                               # FindAndReplaceData('', ''),
                               ]


imaging_devices = ['ct', 'mri']
diagnosis = ['tumor']
locations = ['brain', 'abdomen','stomach','neck', 'lung']

all_tags = list(itertools.chain(imaging_devices,diagnosis,locations ))

dbg_file_csv_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA.csv'
dbg_file_xls_train = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\dumped_data\\vqa_data.xlsx'
dbg_file_xls_processed_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA_post_pre_process.xlsx'
train_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images\\embbeded_images.hdf'

dbg_file_csv_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA.csv'
dbg_file_xls_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA_post_pre_process_intermediate.xlsx'
dbg_file_xls_processed_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA_post_pre_process.xlsx'
validation_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-images\\embbeded_images.hdf'


dbg_file_csv_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA.csv'
dbg_file_xls_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA_post_pre_process_intermediate.xlsx'
dbg_file_xls_processed_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA_post_pre_process.xlsx'
test_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-images\\embbeded_images.hdf'

DataLocations = namedtuple('DataLocations', ['data_tag', 'raw_csv', 'raw_xls', 'processed_xls','embedding'])
train_data = DataLocations('train', dbg_file_csv_train,dbg_file_xls_train,dbg_file_xls_processed_train, train_embedding_path)
validation_data = DataLocations('validation', dbg_file_csv_validation, dbg_file_xls_validation, dbg_file_xls_processed_validation, validation_embedding_path)
test_data = DataLocations('test', dbg_file_csv_test, dbg_file_xls_test, dbg_file_xls_processed_test, test_embedding_path)
