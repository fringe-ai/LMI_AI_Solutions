import os
import argparse
import logging
import json
import numpy as np
import collections
import glob
import pathlib
from label_studio_sdk.converter.brush import decode_rle

from label_utils.csv_utils import write_to_csv
from label_utils.shapes import Rect, Mask, Keypoint, Brush
from label_utils.bbox_utils import convert_from_ls

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LABEL_NAME = 'labels.csv'
PRED_NAME = 'preds.csv'


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        
    def append(self, node):
        self.children.append(node)
        
        
def common_node(paths):
    """find the node that is the common parent of all paths
    """
    root = Node('')
    for path in paths:
        path = pathlib.Path(path).as_posix()
        cur = root
        for part in path.split('/'):
            found = False
            for child in cur.children:
                if child.name == part:
                    cur = child
                    found = True
                    break
            if not found:
                new_node = Node(part)
                cur.append(new_node)
                cur = new_node
    cur = root
    while len(cur.children) == 1:
        cur = cur.children[0]
    return cur


def lst_to_shape(result:dict, fname:str, load_confidence=False):
    """parse the result from label studio result dict, return a Shape object
    """
    result_type=result['type']
    labels = result['value'][result_type]
    if len(labels) > 1:
        raise Exception('Not support more than one labels in a bbox/polygon')
    if len(labels) == 0:
        logger.warning(f'found empty label in {fname}, skip')
        return
    
    label = labels[0]
    conf = result['value']['score'] if load_confidence else 1.0
    if result_type=='rectanglelabels':   
        # get bbox
        x,y,w,h,angle = convert_from_ls(result)
        x1,y1,w,h = list(map(int,[x,y,w,h]))
        x2,y2 = x1+w, y1+h
        rect = Rect(im_name=fname,category=label,up_left=[x1,y1],bottom_right=[x2,y2],angle=angle,confidence=conf)
        return rect
    elif result_type=='polygonlabels':
        points=result['value']['points']
        points_np=np.array(points)
        xs = (points_np[:, 0]/100*result['original_width']).astype(np.int32)
        ys = (points_np[:, 1]/100*result['original_height']).astype(np.int32)
        mask = Mask(im_name=fname,category=label,x_vals=xs,y_vals=ys,confidence=conf)
        return mask
    elif result_type=='brushlabels':
        rle = result['value']['rle']
        h,w = result['original_height'],result['original_width']
        img = decode_rle(rle).reshape(h,w,4)[:,:,3]
        mask = img>128
        brush = Brush(im_name=fname,category=label,mask=mask,confidence=conf)
        return brush
    elif result_type=='keypointlabels':
        dt = result['value']
        x,y = dt['x']/100*result['original_width'],dt['y']/100*result['original_height']
        kp = Keypoint(im_name=fname,category=label,x=x,y=y,confidence=conf)
        return kp
    else:
        logger.warning(f'unsupported result type: {result_type}, skip')


def get_annotations_from_json(path_json):
    """read annotation from label studio json file.

    Args:
        path_json (str): the path to a directory of label studio json files

    Returns:
        dict: a map <image name, a list of Rect objects>
    """
    if os.path.splitext(path_json)[1]=='.json':
        json_files=[path_json]
    else:
        json_files=glob.glob(os.path.join(path_json,'*.json'))
    
    # get common parent node
    for path_json in json_files:
        with open(path_json) as f:    
            l = json.load(f)
        
        img_lists = []
        for dt in l:
            if 'data' not in dt:
                raise Exception('missing "data" in json file. Ensure that the label studio export format is not JSON-MIN.')
            f = dt['data']['image']
            img_lists.append(f)
    root = common_node(img_lists)
    
    annots = {}
    preds = {}
    for path_json in json_files:
        logger.info(f'Extracting labels from: {path_json}')
        with open(path_json) as f:    
            l = json.load(f)
        
        cnt_anno = 0
        cnt_image = 0
        cnt_pred = 0
        cnt_wrong = 0
        prefixes = set(n.name for n in root.children)
        for dt in l:
            f = pathlib.Path(dt['data']['image'])
            l2 = f.as_posix().split('/')
            i = None
            for p in prefixes:
                if p not in l2:
                    continue
                i = l2.index(p)
                key = '_'.join(l2[i:-1])
                
            if key not in annots:
                annots[key] = collections.defaultdict(list)
                preds[key] = collections.defaultdict(list)
                
            fname = f.name
            if fname in annots[key]:
                raise Exception('Found duplicate name')
            
            if 'annotations' in dt:
                cnt = 0
                for annot in dt['annotations']:
                    num_labels = len(annot['result'])
                    if num_labels>0:
                        cnt += 1
                    for result in annot['result']:
                        shape = lst_to_shape(result,fname)
                        if shape is not None:
                            annots[key][fname].append(shape)
                            cnt_anno += 1
                            
                    if 'prediction' in annot and 'result' in annot['prediction']:
                        for result in annot['prediction']['result']:
                            shape = lst_to_shape(result,fname,load_confidence=True)
                            if shape is not None:
                                preds[key][fname].append(shape)
                                cnt_pred += 1
                if cnt>0:
                    cnt_image += 1
                if cnt==0 and dt['total_annotations']>0:
                    cnt_wrong += 1
                    logger.warning(f'found 0 annotation in {fname}, but lst claims total_annotations = {dt["total_annotations"]}')

            if 'predictions' in dt:
                for pred in dt['predictions']:
                    if isinstance(pred, dict):
                        for result in pred['result']:
                            shape = lst_to_shape(result,fname,load_confidence=True)
                            if shape is not None:
                                preds[fname].append(shape)
                                cnt_pred += 1

        logger.info(f'{cnt_image} out of {len(l)} images have annotations')
        if cnt_wrong>0:
            logger.info(f'{cnt_wrong} images with total_annotations > 0, but found 0 annotation')
        logger.info(f'total {cnt_anno} annotations')
        logger.info(f'total {cnt_pred} predictions')
    return annots, preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Convert label studio json file to csv format')
    ap.add_argument('-i', '--path_json', required=True, help='the directory of label-studio json files')
    ap.add_argument('-o', '--path_out', required=True, help='output directory')
    args = ap.parse_args()
    
    annots,preds = get_annotations_from_json(args.path_json)
    
    if os.path.isfile(args.path_out):
        raise Exception('The output path should be a directory')
    
    for k in annots:
        path_out = os.path.join(args.path_out, k)
        os.makedirs(path_out,exist_ok=1)
        
        write_to_csv(annots[k], os.path.join(path_out, LABEL_NAME))
        write_to_csv(preds[k], os.path.join(path_out, PRED_NAME))
