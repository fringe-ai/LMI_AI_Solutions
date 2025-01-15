from detectron2_lmi.model import Detectron2Model
import glob
import os
import cv2
import numpy as np
import json
from label_utils.shapes import Rect, Mask, Keypoint
from label_utils.csv_utils import write_to_csv
import logging

# setup the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def find_images(path:str, exts=['jpg','jpeg','png']):
    """find all images with the given extensions in the path

    Args:
        path (str): the input path
        exts (list): the list of extensions

    Returns:
        list: the list of image paths
    """
    import os
    import glob
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path,f'*.{ext}')))
    return imgs



def inference_run(args):
    model_path = args.get('weights')
    imgs_path = args.get('input')
    out_path = args.get('output')
    class_map_path = args.get('class_map')
    confidence = args.get('confidence')
    
    if not os.path.exists(out_path):
        os.makedirs(args.output)
    
    with open(class_map_path, "r") as f:
        class_map = json.load(f)
        
    
    confidence_map = {
        str(v): confidence for k,v in class_map.items()
    }
    
    # load model
    model = Detectron2Model(model_path, class_map=class_map)
    
    # model warmup
    model.warmup()
    
    # find images
    
    imgs = find_images(imgs_path)
    results = {}
    
    for img_path in imgs:
        csv_results = []
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = model.predict(img, confs=confidence_map, return_segments=True)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        annotated_image = model.annotate_image(
           outputs, img, show_segments=True
        )
        
        # save the image
        fname = os.path.basename(img_path)
        out_img_path = os.path.join(out_path, fname)
        
        cv2.imwrite(out_img_path, annotated_image)
        
        
        # save to csv file
        
        for idx, box in enumerate(outputs['boxes']):
            score = outputs['scores'][idx]
            csv_results.append(
                Rect(im_name=fname, category=outputs['classes'][idx], up_left=box[:2].astype(int).tolist(), bottom_right=box[2:].astype(int).tolist(), confidence=score, angle=0)
            )
            if 'segments' in outputs and len(outputs['segments']) > 0:
                segments = outputs['segments'][idx].astype(int)
                csv_results.append(Mask(im_name=fname, category=outputs['classes'][idx], x_vals=segments[:,0].tolist(), y_vals=segments[:,1].tolist(), confidence=score))
        
        results[fname] = csv_results
    write_to_csv(results, os.path.join(args.output, f"predictions.csv"), overwrite=True)

    
    
    