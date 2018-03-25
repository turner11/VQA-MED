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

dbg_file_xls = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\dumped_data\\vqa_data.xlsx'
dbg_file_csv = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA.csv'
dbg_file_xls_processed = 'C:\\Users\\Public\\Documents\\Data\\2018\\VQAMed2018Train\\VQAMed2018Train-QA_post_pre_process.xlsx'


