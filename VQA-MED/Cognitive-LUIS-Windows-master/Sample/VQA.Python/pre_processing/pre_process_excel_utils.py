import argparse
import openpyxl
import shutil

def find_and_replace(path, text_to_replace, new_text):

    excelFile = openpyxl.load_workbook(path)
    for sheet_name in excelFile.sheetnames:
        sheet1 = excelFile.get_sheet_by_name(sheet_name)
        currentRow = 1
        for row in sheet1.iter_rows():
            for col in row.columns:
                val = str(sheet1.cell(row=currentRow, column=2).value)
                sheet1.cell(row=currentRow, column=col).value = val.replace(text_to_replace, new_text)
            currentRow += 1
        excelFile.save(path)


def process_excel(source, destination):
    shutil.copy(source, destination)
    find_and_replace(destination, text_to_replace, new_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', dest='path', help='path to excel', default='')
    parser.add_argument('-d', dest='destination', help='destination of processed file', default='')

    args = parser.parse_args()
    args.path = args.path or 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\dumped_data\\vqa_data.xlsx'
    args.destination = args.destination or 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\processed_data.xlsx'
    process_excel(args.path,args.destination )


    
