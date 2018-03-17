import argparse
import openpyxl
import shutil

import os

import re

from pre_processing.known_find_and_replace_items import find_and_replace_collection


def find_and_replace(path, find_and_replace_data, columns_filter=None):
    columns_filter = columns_filter or (lambda clm_name: True)

    excelFile = openpyxl.load_workbook(path)
    for sheet_name in excelFile.sheetnames:
        curr_sheet = excelFile.get_sheet_by_name(sheet_name)


        cels_in_first_row = list(curr_sheet.iter_rows())[0]
        cols = [(c.value, idx) for c, idx in zip(cels_in_first_row, range(len(cels_in_first_row)))]
        cols_idxs = [c[1] for c in cols if columns_filter(c[0])]

        current_row = 1
        for row in curr_sheet.iter_rows():
            for col_idx in range(len(row)):
                if col_idx not in cols_idxs:
                    continue
                seed_one_col = col_idx+1
                curr_cell =curr_sheet.cell(row=current_row, column=seed_one_col)
                val = str(curr_cell.value)
                if not val or val == 'None':
                    continue
                new_val = val
                for tpl in find_and_replace_data:
                    pattern = re.compile(tpl.orig, re.IGNORECASE)
                    new_val = pattern.sub(tpl.sub, new_val)
                    # new_val = new_val.replace(tpl.orig, tpl.sub)
                curr_cell.value = new_val.strip()
            current_row += 1
    print(path)
    excelFile.save(path)


def process_excel(source, destination):
    shutil.copy(source, destination)
    find_and_replace_data = find_and_replace_collection
    TOLENIZED_COL_PREFIX = 'tokenized_'
    col_filter = lambda col_name: TOLENIZED_COL_PREFIX in (col_name or '')
    find_and_replace(destination, find_and_replace_data, columns_filter=col_filter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', dest='path', help='path to excel', default='')
    parser.add_argument('-d', dest='destination', help='destination of processed file', default='')

    args = parser.parse_args()
    args.path = args.path or 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\dumped_data\\vqa_data.xlsx'
    args.destination = args.destination or os.path.splitext(args.path )[0] + '_post_pre_process.xlsx'
    process_excel(args.path,args.destination)


    
