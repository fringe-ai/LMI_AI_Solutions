
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

def crop_by_percent(image, crop_percent, crop_from = 'top'):
    height, width = image.shape[:2]
    x1, y1 = 0, 0
    x2, y2 = width, height
    if crop_from == 'top':
        x1, y1 = 0, 0
        x2, y2 = width, int(height * crop_percent)
    elif crop_from == 'bottom':
        x1, y1 = 0, int(height * (1 - crop_percent))
        x2, y2 = width, height
    return x1, y1, x2, y2

def crop_kp(bbox, shape):
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    x = shape.x - x1
    y = shape.y - y1
    
    
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
    ap.add_argument('--crop_by_percent', type=float, default=0.0, help='the percentage of the image to crop', required=False)
    ap.add_argument('--crop_from', default='top', help='the direction to crop the image', required=False)
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
        
        if args['crop_by_percent']>0:
            x1,y1,x2,y2 = foreground_shapes[fname]['foreground']
            sx,sy,ex,ey = crop_by_percent(image[y1:y2, x1:x2], args['crop_by_percent'], args['crop_from'])
            logger.info(f'cropping {fname} by {args["crop_by_percent"]*100:.2f}% from {args["crop_from"]}')
        
            
        
        H,W = image.shape[:2]
        for shape in fname_to_shapes[fname]:
            
            if shape.category in target_classes:
                continue
            
            
            if isinstance(shape, Rect):
                x1,y1 = shape.up_left
                x2,y2 = shape.bottom_right
                bbox = [x1,y1,x2,y2]
                cropped_bbox = crop_bbox(foreground_shapes[fname]['foreground'], bbox)
                if args['crop_by_percent']>0:
                    cropped_bbox = crop_bbox([sx,sy,ex,ey], cropped_bbox)
                annots[updated_fname].append(
                     Rect(
                        up_left=[cropped_bbox[0], cropped_bbox[1]],
                        bottom_right=[cropped_bbox[2], cropped_bbox[3]],
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
                if args['crop_by_percent']>0:
                    cropped_mask, _ = crop_mask([sx,sy,ex,ey], cropped_mask, bbox_format="xyxy")
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
                if args['crop_by_percent']>0:
                    x,y,is_valid = crop_kp([sx,sy,ex,ey], Keypoint(x=x, y=y, confidence=shape.confidence, category=shape.category, im_name=updated_fname, fullpath=os.path.join(args['path_out'], updated_fname)))
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
            
            # suporting polygon mask
            if isinstance(shape, Mask):
                polygon_x = shape.X
                polygon_y = shape.Y
                polygon_mask = np.stack([polygon_x, polygon_y], axis=1)
                _, cropped_polygon = crop_mask(foreground_shapes[fname]['foreground'], mask=None, polygon_mask=polygon_mask, bbox_format="xyxy")
                if args['crop_by_percent']>0:
                    _, cropped_polygon = crop_mask([sx,sy,ex,ey], mask=None, polygon_mask=cropped_polygon, bbox_format="xyxy")
                # add to mask labels
                annots[updated_fname].append(
                    Mask(
                        x_vals=cropped_polygon[:, 0].tolist(),
                        y_vals=cropped_polygon[:, 1].tolist(),
                        confidence=shape.confidence,
                        category=shape.category,
                        im_name=updated_fname,
                        fullpath=os.path.join(args['path_out'], updated_fname)
                    )
                )
                
        # save the cropped image
        x1,y1,x2,y2 = foreground_shapes[fname]['foreground']
        cropped_image = image[y1:y2, x1:x2]
        if args['crop_by_percent']>0:
            cropped_image = cropped_image[sy:ey, sx:ex]
        
        cv2.imwrite(os.path.join(args['path_out'], updated_fname), cropped_image)
        
    
    # save the updated shapes
    write_to_csv(annots, os.path.join(args['path_out'], 'labels.csv'))
                
                    

if __name__ == '__main__':
    main()
                    
                    
