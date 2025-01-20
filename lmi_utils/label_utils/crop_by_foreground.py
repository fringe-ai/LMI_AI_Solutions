
import numpy as np
from label_utils.csv_utils import load_csv
from label_utils.shapes import Mask, Rect, Keypoint, Brush
from label_utils.csv_utils import write_to_csv
import os
import cv2
import collections
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def crop_kp(bbox, shape):
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    x,y = shape.x, shape.y
    x -= x1
    y -= y1
    
    valid = True
    if x<0 or x>=w or y<0 or y>=h:
        valid = False
        logger.warning(f'in {shape.im_name}, keypoint {x:.4f},{y:.4f} is out of the foreground bbox: {x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}. skip')
    
    return x,y,valid


def crop_bbox(bbox1, bbox2):
    crop_x1, crop_y1, crop_x2, crop_y2 = bbox1
    target_x1, target_y1, target_x2, target_y2 = bbox2
    adjusted_x1 = target_x1 - crop_x1
    adjusted_y1 = target_y1 - crop_y1
    adjusted_x2 = target_x2 - crop_x1
    adjusted_y2 = target_y2 - crop_y1
    return adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2

def crop_mask(bbox, mask=None, polygon_mask=None, bbox_format="xywh"):
    # Interpret the bounding box coordinates
    if bbox_format == "xywh":
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
    elif bbox_format == "xyxy":
        xmin, ymin, xmax, ymax = bbox
        w, h = xmax - xmin, ymax - ymin
    else:
        raise ValueError("bbox_format must be either 'xywh' or 'xyxy'")
    cropped_mask = None
    if mask is not None:
        cropped_mask = mask[ymin:ymax, xmin:xmax]
    
    # If a polygon mask is provided, adjust its coordinates relative to the crop.
    cropped_polygon = None
    if polygon_mask is not None:
        # Ensure the input is a numpy array of shape (N, 2)
        polygon_mask = np.asarray(polygon_mask)
        if polygon_mask.ndim != 2 or polygon_mask.shape[1] != 2:
            raise ValueError("polygon_mask must be a 2D array with shape (N_points, 2)")
        
        # Shift the polygon by the top-left corner of the bounding box.
        cropped_polygon = polygon_mask - np.array([xmin, ymin])
        
        cropped_polygon[:, 0] = np.clip(cropped_polygon[:, 0], 0, w)
        cropped_polygon[:, 1] = np.clip(cropped_polygon[:, 1], 0, h)
    
    return cropped_mask, cropped_polygon


def main():
    import argparse
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--path_imgs', '-i', required=True, help='the path of a image folder')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--path_out', '-o', required=True, help='the output path')
    ap.add_argument('--target_classes',required=True, help='the comma separated target classes to crop')
    args = vars(ap.parse_args())
    path_imgs = args['path_imgs']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    target_classes = args['target_classes'].split(',')
    fname_to_shapes,class_to_id = load_csv(path_csv, path_imgs, zero_index=True)
    if not os.path.exists(args['path_out']):
        os.makedirs(args['path_out'])
    
    foreground_shapes = {}
    for fname in fname_to_shapes:
        if fname not in foreground_shapes:
            foreground_shapes[fname] = {
                'foreground': []
            }
        for shape in fname_to_shapes[fname]:
            #get class ID
            if shape.category not in target_classes:
                continue
            
            
            if isinstance(shape, Rect):
                x0,y0 = shape.up_left
                x2,y2 = shape.bottom_right
                foreground_shapes[fname]['foreground'] = list(map(int, [x0,y0,x2,y2]))
        if len(foreground_shapes[fname]['foreground'])==0:
            logger.warning(f'no foreground found in {fname}')
            
                
    annots = collections.defaultdict(list)
    for fname in fname_to_shapes:
        ext = os.path.basename(fname).split('.')[-1]
        updated_fname = os.path.basename(fname).replace(f'.{ext}', f'_cropped.{ext}')
        if fname not in foreground_shapes:
            continue
        if len(foreground_shapes[fname]['foreground'])==0:
            continue
        logger.info(f'processing {fname}')
        
        image = cv2.imread(os.path.join(path_imgs, fname))
        if image is None:
            logger.warning(f'failed to read {fname}, skip')
            continue
        
        H,W = image.shape[:2]
        for shape in fname_to_shapes[fname]:
            
            if shape.category in target_classes:
                continue
            
            
            if isinstance(shape, Rect):
                x1,y1 = shape.up_left
                x2,y2 = shape.bottom_right
                bbox = [x1,y1,x2,y2]
                updated_bbox = crop_bbox(foreground_shapes[fname]['foreground'], bbox)
                annots[updated_fname].append(
                     Rect(
                        up_left=[updated_bbox[0], updated_bbox[1]],
                        bottom_right=[updated_bbox[2], updated_bbox[3]],
                        angle=shape.angle,
                        confidence=shape.confidence,
                        category=shape.category,
                        im_name=updated_fname,
                        fullpath=os.path.join(args['path_out'], updated_fname)
                    )
                )
            
            if isinstance(shape, Brush):
                mask = shape.to_mask((H,W))
                mask = mask.astype(np.uint8)*255
                # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cropped_mask, _ = crop_mask(foreground_shapes[fname]['foreground'], mask, bbox_format="xyxy")
                # add to brush labels
                annots[updated_fname].append(
                    Brush(
                        mask=cropped_mask>128,
                        confidence=shape.confidence,
                        category=shape.category,
                        im_name=updated_fname,
                        fullpath=os.path.join(args['path_out'], updated_fname)
                    )
                )
                
            if isinstance(shape, Keypoint):
                x,y,is_valid = crop_kp(foreground_shapes[fname]['foreground'], shape)
                if not is_valid:
                    continue
                annots[updated_fname].append(
                    Keypoint(
                        x=x,
                        y=y,
                        confidence=shape.confidence,
                        category=shape.category,
                        im_name=updated_fname,
                        fullpath=os.path.join(args['path_out'], updated_fname)
                    )
                )
                
                
        # save the cropped image
        x1,y1,x2,y2 = foreground_shapes[fname]['foreground']
        cropped_image = image[y1:y2, x1:x2]
        
        cv2.imwrite(os.path.join(args['path_out'], updated_fname), cropped_image)
        
    
    # save the updated shapes
    write_to_csv(annots, os.path.join(args['path_out'], 'labels.csv'))
                
                    

if __name__ == '__main__':
    main()
                    
                    
