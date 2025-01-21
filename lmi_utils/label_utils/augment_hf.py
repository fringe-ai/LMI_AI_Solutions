
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--path_imgs', '-i', required=True, help='the path of a image folder')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--path_out', '-o', required=True, help='the output path')
    ap.add_argument('--symc',required=True, help='the comma separated symetrical classes. Every two classes are symetrical to each other.')
    args = vars(ap.parse_args())
    path_imgs = args['path_imgs']
    symc = args['symc'].split(',')
    if len(symc) % 2 != 0:
        raise ValueError('symc should have even number of elements')
    symc_dict = {}
    for i in range(0, len(symc), 2):
        symc_dict[symc[i]] = symc[i+1]
        symc_dict[symc[i+1]] = symc[i]
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    fname_to_shapes,class_to_id = load_csv(path_csv, path_imgs, zero_index=True)
    if not os.path.exists(args['path_out']):
        os.makedirs(args['path_out'])
            
                
    annots = fname_to_shapes.copy()
    for fname in fname_to_shapes:
        logger.info(f'processing {fname}')
        ext = os.path.basename(fname).split('.')[-1]
        updated_fname = os.path.basename(fname).replace(f'.{ext}', f'_augmented_hf.{ext}')
        img = cv2.imread(os.path.join(args['path_imgs'], fname))
        height, width = img.shape[:2]
        flipped_image = img[:, ::-1].copy()
        logger.info(f'flipped image shape: {flipped_image.shape}')
        
        for shape in fname_to_shapes[fname]:
            new_symc = shape.category
            if shape.category in symc_dict:
                new_symc = symc_dict[shape.category]
                
            if isinstance(shape, Rect):
                x1,y1 = shape.up_left
                x2,y2 = shape.bottom_right
                bbox = [x1,y1,x2,y2]
                
                flipped_bbox = [width - x2, y1, width - x1, y2]
                annots[updated_fname].append(Rect(
                        up_left=[flipped_bbox[0], flipped_bbox[1]],
                        bottom_right=[flipped_bbox[2], flipped_bbox[3]],
                        angle=shape.angle,
                        confidence=shape.confidence,
                        category=shape.category,
                        im_name=updated_fname,
                        fullpath=os.path.join(args['path_out'], updated_fname)
                    ))
            elif isinstance(shape, Keypoint):
                
                # flip the keypoint
                x = shape.x
                y = shape.y
                
                flipped_x = width - x
                flipped_y = y
                annots[updated_fname].append(Keypoint(
                        x=flipped_x,
                        y=flipped_y,
                        confidence=shape.confidence,
                        category=new_symc,
                        im_name=updated_fname,
                        fullpath=os.path.join(args['path_out'], updated_fname)
                ))
        cv2.imwrite(os.path.join(args['path_out'], fname), img)
        cv2.imwrite(os.path.join(args['path_out'], updated_fname), flipped_image)
    
    # write the updated shapes to a csv file
    
    write_to_csv(annots, os.path.join(args['path_out'], 'labels.csv'))
    logger.info('done augmenting hf')
                
                    

if __name__ == '__main__':
    main()
                    
                    
