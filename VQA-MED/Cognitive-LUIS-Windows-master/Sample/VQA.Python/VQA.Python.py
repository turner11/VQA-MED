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
class VqaDataParser(object):
    
    def __init__(self, json_path, **kwargs):
        if not json_path or not os.path.isfile(json_path):
            raise Exception("Got a non valid path: {0}".format(json_path))
        
        self.json_path = json_path
        with open(json_path) as f:
            self.json_data = json.load(f)
        return super().__init__(**kwargs)

    def __repr__(self, **kwargs):
        return "{0}({1})".format(self.__class__.__name__,self.json_path)


    def get_all_data(self, query=None):
        json_data = self.json_data
        
        annotations = json_data["annotations"]   
        images = json_data["images"]
        categories = json_data["categories"]

        if query:
            # Query images:
            img_query_fields = ['file_name','id']
            images_by_images_id = list((img['id'], img) for img in images if any([query in str(img[k]) for k in img_query_fields]))

            # Query annotations:
            an_query_fields = ['image_id','id']
            image_an = list((an['image_id'], an) for an in annotations if any([query in str(an[k]) for k in an_query_fields]))
            
            # Query categories:
            cat_query_fields = ['name','supercategory',"id"]
            cat_ids = [c['id'] for c in categories if any([query in str(c[k]) for k in cat_query_fields])]
            relevant_images_with_category = set(an['image_id'] for an in annotations if an['category_id'] in cat_ids)

            
            relevant_img_ids = set([tpl[0] for tpl in image_an]).union(set([tpl[0] for tpl in images_by_images_id])).union(relevant_images_with_category )
            relevant_img_ids = sorted(relevant_img_ids)
            
            filename_by_id = { img_data['id']:img_data['file_name'] for img_data in images if img_data['id'] in relevant_img_ids}

            images = [img for img in images if img['id'] in relevant_img_ids]
            annotations = [an for an in annotations if an['image_id'] in relevant_img_ids]
            all_relevant_categories = [an[ 'category_id'] for an in annotations]
            categories = [c for c in categories if c['id']in all_relevant_categories]
        
            #mutual_keys = relevant_ids.intersection(set(filename_by_id.keys()))
            #q_results = {filename_by_id[id]: captions_by_id[id] for id in mutual_keys}

        return images, annotations, categories


    def get_image_data_items(self, image_name=None, image_path=None):
        if not image_name and not image_path:
            raise Exception("image name or image path must be specified")

        if not image_name:
            image_folder, image_name = os.path.split(image_path)

        images, annotations, categories = self.get_all_data()
    
        #image_id = int(os.path.basename(image_path).split("_")[-1].split(".")[0])
        image_id = int(image_name.split(".")[0])

        image_id, image_info = next((i.get('id',-1),i) for i in images if i['file_name'] == image_name)
        if image_id<0:
             raise Exception("Got an image without id ({0})".format(image_id))
        
        relevant_annotations = [a for a in annotations if a.get('image_id',-2)== image_id]
        


        categories_ids = [a.get('category_id',-3) for a in relevant_annotations]
        relevant_ctegories = [c for c in categories if c.get('id',-4) in categories_ids]
       

        return image_info, relevant_annotations, relevant_ctegories


    def get_image_data(self, image_path):    
        image_info, annotations, categories = self.get_image_data_items(image_path=image_path) 
    
        caption = ""
        ret_image_info = {}
        errors = []
        try:
            annotations_info = ["\n".join(["{0}: {1}".format(k,v) for k,v in a.items()]) for a in annotations]
            #just for nicer display...
            keys = set(list(itertools.chain.from_iterable([list(c.keys()) for c in categories])))
            keys = sorted(keys,key=lambda k: len(str(k)))
            ordered_categories = []
            for c in categories:
                ordered = OrderedDict()
                ordered_categories.append(ordered)
                for k in keys:
                    ordered[k] = c[k]
            
                        

            categories_info = ["; ".join(["{0}: {1}".format(k,v).replace("\n","; ") for k,v in c.items()]) for c in ordered_categories]
            logger.debug(image_path)
            ret_image_info.update(image_info)
            ret_image_info['Meta'] = "{0} annotations\n{1} categories".format(len(annotations_info), len(categories_info))
            ret_image_info['annotations'] = annotations_info
            ret_image_info['categories'] = categories_info
            #image_info = next((im_data for im_data in images_dict if im_data.get('file_name',"NO-NAME") == image_name),{})
            #logger.debug("image info was: {0}".format(image_info))
            ##caption = ";\n\n".join([an[CAPTION_KEY] for an in image_an]) or "FAILED TO GET CAPTION"
            #caption = [an[CAPTION_KEY] for an in image_an] or ["FAILED TO GET CAPTION"]
        except Exception as ex:
            logger.warn("@@@@@@@@@@@@@ Error @@@@@@@@@:\n{0}".format(ex))
            #caption = caption or "Failed to get caption"
            ret_image_info = ret_image_info or {}
            errors.append("Got an exception: {0}".format(ex))
        
        if errors:
            ret_image_info[ERROR_KEY] = errors
    

        #image_info.update({CAPTION_KEY:caption})    
        return ret_image_info


    def query_data(self,query):    
        images, annotations, categories = self.get_all_data(query=query) 
    
        caption = ""
        q_results = {}
        try:            
            #q_results = { d['id']:d['file_name'] for d in images}
            #q_results = {"query_results":[d['file_name'] for d in images]}
            q_results = { d['file_name']:d['id'] for d in images}
        except Exception as ex:        
            q_results = q_results or {}
            q_results[ERROR_KEY] = str(ex)

        return q_results
        

def main(args):
    try:
    
        # args.path =  "C:\\Users\\Public\\Documents\\Data\\2014 Train\\annotations\\captions_train2014.json"
        #args.imag_name = "COCO_train2014_000000318495.jpg"

        #args.imag_name = None
        #args.query = 'amazing'#'zebra'
        file_name = args.path
        image_name = args.imag_name
        query = args.query
        
        parser = VqaDataParser(file_name)
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
        import traceback
        tb = traceback.format_exc()       
        ret_val = {ERROR_KEY: "Got an unexpected error:\n{0}\n\nTraceback:\n{1}".format(ex,tb)}
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


    #pix_p = "C:\\Users\\Public\\Documents\\Data\\2017\\annotations\\stuff_val2017_pixelmaps"
    #imgs_p = "C:\\Users\\Public\\Documents\\Data\\2017\\val2017"

    #def clean_names(path):
    #    return sorted([int(fn.split(".")[0]) for fn in os.listdir(path)])
    #pix = clean_names(pix_p)
    #img = clean_names(imgs_p)
    #inter = set(pix).intersection(set(img))
    #print(inter)



    #args.path = "C:\\Users\\Public\\Documents\\Data\\2017\\annotations\\stuff_val2017.json"
    #args.imag_name= "C:\\Users\\Public\\Documents\\Data\\2017\\val2017\\000000000285.jpg"#"C:\\Users\\Public\\Documents\\Data\\2017\\val2017\\000000000139.jpg"#"C:\\Users\\Public\\Documents\\Data\\2017\\val2017\\000000001584.jpg"
    
    #args.path = "C:\\Users\\Public\\Documents\\Data\\2017\\annotations\\stuff_val2017.json"
    #args.imag_name= None
    #args.query = "13"
    
    
    ret_val = main(args)
    json_string = json.dumps(ret_val)
    print(json_string)
        #raw_input()

    #sys.exit(int(main(file_name) or 0))

    #python C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
    #"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py" -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"