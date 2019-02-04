from collections import namedtuple
import itertools

FindAndReplaceData = namedtuple('FindAndReplaceData', ['orig', 'sub'])


remove_stop_pattern = r'\b(a |an |the |in |of |)*\b{0}\b( scan| image)*\b'
remove_stop  = [FindAndReplaceData(remove_stop_pattern.format(device), device) for device in ['ct', 'mri', 'mra', 'cta']]
find_and_replace_collection = [FindAndReplaceData(r'magnetic resonance imaging', 'mr'),
                               FindAndReplaceData(r'magnetic resonance angiography', 'mr'),
                               FindAndReplaceData(r'mri', 'mr'),
                               ]+remove_stop

imaging_devices = ['ct', 'mr', 'us', 'xr', 'angiogram', 'mammograph']
diagnosis = ['fracture',
             'cyst',
             'cerebral',
             'acute',
             'syndrome',
             'cell',
             'disease',
             'artery',
             'carcinoma',
             'malformation',
             'tumor',
             'pulmonary',
             'lymphoma',
             'venous',
             'abscess',
             'meningioma',
             'aortic',
             'sclerosis',
             'astrocytoma',
             'spinal',
             'infarction',
             'renal',
             'multiple',
             'glioblastoma',
             'multiforme',
             'aneurysm',
             'thrombosis',
             'intracranial',
             'arteriovenous',
             'posterior',
             'adenocarcinoma',
             'bone',
             'dural',
             'secondary',
             'schwannoma',
             'nerve',
             'cancer',
             'diffuse',
             'carotid',
             'sinus',
             'central',
             'metastatic',
             'cerebellar',
             'appendicitis',
             'kidney',
             'hernia',
             'epidermoid',
             'infarct',
             'lung',
             'orbital',
             'glioma',
             'histiocytosis',
             'vein',
             'dysplasia',
             'arachnoid',
             'subclavian',
             'hemangioma',
             'cavernous',
             'cord',
             'breast',
             'epidural',
             'dermoid',
             'tear',
             'osteomyelitis',
             'lipoma',
             'anterior',
             'dissection',
             'hemorrhage',
             'esophageal',
             'embolism',
             'ependymoma',
             'hematoma',
             'adenoma',
             'disseminated',
             'encephalomyelitis',
             'tuberous',
             'medulloblastoma',
             'b-cell',
             'injury',
             'arch']

locations = ['lung, mediastinum, pleura',
             'skull and contents',
             'genitourinary',
             'spine and contents',
             'musculoskeletal',
             'heart and great vessels',
             'vascular and lymphatic',
             'gastrointestinal',
             'face, sinuses, and neck',
             'breast']

planes = ['axial',
          'longitudinal',
          'coronal',
          'lateral',
          'ap',
          'sagittal',
          'mammo - mlo',
          'pa',
          'mammo - cc',
          'transverse',
          'mammo - mag cc',
          'frontal',
          'oblique',
          '3d reconstruction',
          'decubitus']

all_tags = list(itertools.chain(imaging_devices, diagnosis, locations, planes))

models_folder = "C:\\Users\\Public\\Documents\\Data\\2018\\models"

dbg_file_csv_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA.csv'
dbg_file_xls_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA_post_pre_process_intermediate.xlsx'  # "'C:\\\\Users\\\\avitu\\\\Documents\\\\GitHub\\\\VQA-MED\\\\VQA-MED\\\\Cognitive-LUIS-Windows-master\\\\Sample\\\\VQA.Python\\\\dumped_data\\\\vqa_data.xlsx'
dbg_file_xls_processed_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA_post_pre_process.xlsx'
train_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images\\embbeded_images.hdf'
images_path_train = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-images'

dbg_file_csv_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA.csv'
dbg_file_xls_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA_post_pre_process_intermediate.xlsx'
dbg_file_xls_processed_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-QA_post_pre_process.xlsx'
validation_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-images\\embbeded_images.hdf'
images_path_validation = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Valid\\VQAMed2018Valid-images'

dbg_file_csv_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA.csv'
dbg_file_xls_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA_post_pre_process_intermediate.xlsx'
dbg_file_xls_processed_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-QA_post_pre_process.xlsx'
test_embedding_path = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-images\\embbeded_images.hdf'
images_path_test = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Test\\VQAMed2018Test-images'

DataLocations = namedtuple('DataLocations', ['data_tag', 'raw_csv', 'raw_xls', 'processed_xls', 'images_path'])
train_data = DataLocations('train', dbg_file_csv_train, dbg_file_xls_train, dbg_file_xls_processed_train,
                           images_path_train)
validation_data = DataLocations('validation', dbg_file_csv_validation, dbg_file_xls_validation,
                                dbg_file_xls_processed_validation, images_path_validation)
test_data = DataLocations('test', dbg_file_csv_test, dbg_file_xls_test, dbg_file_xls_processed_test, images_path_test)
