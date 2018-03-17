from collections import namedtuple

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


                               # FindAndReplaceData('', ''),
                               ]


#