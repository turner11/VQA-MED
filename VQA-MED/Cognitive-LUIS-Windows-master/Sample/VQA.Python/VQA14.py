import json
import itertools
import sys
import argparse
import os
import logging 
from collections import OrderedDict

ERROR_KEY = "error"
logname = "vqa.log"
logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)



logger = logging.getLogger('pythonVQA')


CAPTION_KEY = 'caption'
class Vqa14DataParser(object):
    def __init__(self, json_path, **kwargs):
        if not json_path or not os.path.isfile(json_path):
            raise Exception("Got a non valid path: {0}".format(json_path))
        
        self.json_path = json_path
        with open(json_path) as f:
            self.json_data = json.load(f)
        super().__init__(**kwargs)
    
    def __repr__(self, **kwargs):
        return "{0}({1})".format(self.__class__.__name__,self.json_path)


    def get_all_data(self, query=None):
        json_data = self.json_data
        
        annotations = json_data["annotations"]   
        images = json_data["images"]
        #categories = csv_data["categories"]

        if query:
            # Query images:
            img_query_fields = ['file_name','id']
            images_by_images_id = list((img['id'], img) for img in images if any([query in str(img[k]) for k in img_query_fields]))

            # Query annotations:
            an_query_fields = ['image_id','id','caption']
            image_an = list((an['image_id'], an) for an in annotations if any([query in str(an[k]) for k in an_query_fields]))
            
            ## Query categories:
            #cat_query_fields = ['name','supercategory',"id"]
            #cat_ids = [c['id'] for c in categories if any([query in str(c[k]) for k in cat_query_fields])]
            #relevant_images_with_category = set(an['image_id'] for an in annotations if an['category_id'] in cat_ids)

            
            relevant_img_ids = set([tpl[0] for tpl in image_an]).union(set([tpl[0] for tpl in images_by_images_id]))#.union(relevant_images_with_category )
            relevant_img_ids = sorted(relevant_img_ids)
            
            filename_by_id = { img_data['id']:img_data['file_name'] for img_data in images if img_data['id'] in relevant_img_ids}

            images = [img for img in images if img['id'] in relevant_img_ids]
            annotations = [an for an in annotations if an['image_id'] in relevant_img_ids]
            #all_relevant_categories = [an[ 'category_id'] for an in annotations]
            #categories = [c for c in categories if c['id']in all_relevant_categories]
        
            #mutual_keys = relevant_ids.intersection(set(filename_by_id.keys()))
            #q_results = {filename_by_id[id]: captions_by_id[id] for id in mutual_keys}

        return images, annotations#, categories
    

    def get_image_data(self, image_path):    
        images_dict, annot_dict = self.get_all_data() 
    
        caption = ""
        image_info = {}
        try:
            image_id = int(os.path.basename(image_path).split("_")[-1].split(".")[0])
            image_folder, image_name = os.path.split(image_path)
            image_an = [an for an in annot_dict if an.get('image_id',-999) == image_id]
         
            logger.debug(image_path)
            image_info = next((im_data for im_data in images_dict if im_data.get('file_name',"NO-NAME") == image_name),{})
            logger.debug("image info was: {0}".format(image_info))
            #caption = ";\n\n".join([an[CAPTION_KEY] for an in image_an]) or "FAILED TO GET CAPTION"
            caption = [an[CAPTION_KEY] for an in image_an] or ["FAILED TO GET CAPTION"]
        except Exception as ex:
            logger.warn("@@@@@@@@@@@@@ Error @@@@@@@@@:\n{0}".format(ex))
            caption = caption or "Failed to get caption"
            image_info = image_info or {}
            image_info[ERROR_KEY] = str(ex)

        image_info.update({CAPTION_KEY:caption})    
        return image_info


    def query_data(self, query):    
        images_dict, annot_dict = self.get_all_data(query=query) 
    
        caption = ""
        q_results = {}
        try:                
            image_an = (an for an in annot_dict if query in an.get(CAPTION_KEY,""))
            captions_by_id = {an['image_id']: an[CAPTION_KEY] for an in image_an}

            relevant_ids = set(captions_by_id.keys())
            filename_by_id = { d['id']:d['file_name'] for d in images_dict if d['id'] in relevant_ids}
        
            mutual_keys = relevant_ids.intersection(set(filename_by_id.keys()))
            q_results = {filename_by_id[id]: captions_by_id[id] for id in mutual_keys}
            #image_info = next((im_data for im_data in images_dict if im_data.get('file_name',"NO-NAME") == image_name),{})
            #logger.debug("image info was: {0}".format(image_info))
            #caption = image_an['caption']        
        except Exception as ex:        
            q_results = q_results or {}
            q_results[ERROR_KEY] = str(ex)

        return q_results
        

def main(args):
    try:
    
        #args.path =  "C:\\Users\\Public\\Documents\\Data\\2014 Train\\annotations\\captions_train2014.json"
        #args.imag_name = "COCO_train2014_000000318495.jpg"

        #args.imag_name = None
        #args.query = 'amazing'#'zebra'

        file_name = args.path
        image_name = args.imag_name

        parser = Vqa14DataParser(file_name)
        query = args.query       
        if args.imag_name:
            image_info = parser.get_image_data(image_name)        
            ret_val = image_info
        elif query:
            ret_val = parser.query_data(query)
            #ret_val = {"Testing": "query was: {0}".format(query)}
        elif args.question:
            ret_val = {ERROR_KEY: "Asking a question was not implemented yet..."}
        else:
             ret_val = {ERROR_KEY: "Could not figure out what you want me to do..."}
    except Exception as ex:
         ret_val = {ERROR_KEY: "Got a non expected error: {0}".format(ex)}
    finally: 
        pass
    return ret_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts caption for a COCO image.')    
    parser.add_argument('-p', dest='path', help='path of annotation file')
    parser.add_argument('-n', dest='imag_name', help='name_of_image',default=None)
    parser.add_argument('-q', dest='query', help='query to look for',default=None)
    parser.add_argument('-a', dest='question', help='question to ask',default=None)

    args = parser.parse_args()  

    ret_val = main(args)
    json_string = json.dumps(ret_val)
    print(json_string)
        #raw_input()

    #sys.exit(int(main(file_name) or 0))

    #python C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
    #"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py" -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
