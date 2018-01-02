import json
import sys
import argparse
import os


def get_caption(file_name,image_name ):    
    with open(file_name) as f:
        t = json.load(f)
    #t.keys()
    a = t["annotations"]   
   
    try:
        image_id = int(os.path.basename(image_name).split("_")[-1].split(".")[0])
        image_an = next(an for an in a if an['image_id'] == image_id)
        caption = image_an['caption']
    except Exception as ex:
        caption = "Failed to get caption"

    return caption
        

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Extracts caption for a COCO image.')    
        parser.add_argument('-p', dest='path', help='path of annotation file')
        parser.add_argument('-n', dest='imag_name', help='name_of_image')

        args = parser.parse_args()

        file_name = args.path
        image_name = args.imag_name
    
        #file_name =  "C:\\Users\\Public\\Documents\\Data\\2014 Train\\annotations\\captions_train2014.json"
        #image_name = "COCO_train2014_000000318495.jpg"


        caption = get_caption(file_name,image_name)
        print(caption)
    finally:
        raw_input()

    #sys.exit(int(main(file_name) or 0))

