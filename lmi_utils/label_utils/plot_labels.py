import numpy as np
import random
import cv2
import os
import json
import logging

#LMI packages
# from label_utils.shapes import Rect, Mask, Keypoint, Brush
from dataset_utils.representations import Dataset, Annotation, FileAnnotations, MaskType, Mask,AnnotationType
from label_utils.plot_utils import plot_one_box, plot_one_polygon, plot_one_pt, plot_one_brush
from label_utils.bbox_utils import rotate


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_shape(shape, im, color_map):
    if shape.type == AnnotationType.BOX:
        x1,y1, x2, y2, angle = shape.value.coords()
        width = x2 - x1
        height = y2 - y1
        # rotated rectangle
        if angle > 0:
            rotated_rect = rotate(x1, y1, width, height, angle)
        else:
            rotated_rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        plot_one_polygon(np.array([rotated_rect]), im, label=shape.label_id, color=color_map[shape.label_id])
    elif shape.type == AnnotationType.MASK:
        # pts = np.array([[x,y] for x,y in zip(shape.X,shape.Y)])
        # pts = pts.reshape((-1, 1, 2)).astype(int)
        
        x,y = shape.value.coords()
        if shape.value.type == MaskType.POLYGON:
            pts = np.array([[x,y]]).reshape((-1, 1, 2)).astype(int)
            plot_one_polygon(pts, im, label=shape.label_id, color=color_map[shape.label_id])
        elif shape.value.type == MaskType.BITMASK:
            x,y = shape.value.coords()
            plot_one_brush(x,y,im,label=shape.label_id,color=color_map[shape.label_id])
    elif shape.type == AnnotationType.KEYPOINT:
        x,y,_ = shape.value.coords()
        plot_one_pt([x,y], im, label=shape.label_id, color=color_map[shape.label_id])
    else:
        raise Exception(f'Unknown shape: {type(shape)}')
    return



if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--path_imgs', required=True, help='the path to the input image folder')
    ap.add_argument('-o','--path_out', required=True, help='the path to the output folder')
    ap.add_argument('--path_csv', default='labels.json', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.json" in path_imgs')
    ap.add_argument('--class_map_json', default=None, help='[optinal] the path of a class map json file')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_csv = args['path_csv'] if args['path_csv']!='labels.json' else os.path.join(path_imgs, args['path_csv'])
    output_path = args['path_out']
    # if args['class_map_json']:
    #     with open(args['class_map_json']) as f:
    #         class_map = json.load(f)
    #     logger.info(f'loaded class map: {class_map}')
    # else:
    #     class_map = None

    # if not os.path.isfile(path_csv):
    #     raise Exception(f'Not found file: {path_csv}')
    assert path_imgs!=output_path, 'output path must be different with input path'
    # fname_to_shape, class_map = load_csv(path_csv, path_imgs, class_map)
    
    dataset = Dataset.load(path_csv)
    base_prefix = dataset.base_path
    
    # init color map
    color_map = {}
    for cls in dataset.get_labels():
        logger.info(f'CLASS: {cls}')
        color_map[cls] = tuple([random.randint(0,255) for _ in range(3)])
    os.makedirs(output_path, exist_ok=True)
    
    for f in dataset.files:
        file_path = f.relative_path(base_prefix)
        im_name = os.path.basename(file_path)
        logger.info(f'processing {im_name}')
        if not os.path.exists(os.path.join(path_imgs, file_path)):
            raise Exception(f'file not found: {file_path}')
        im = cv2.imread(os.path.join(path_imgs, file_path))
        h,w = im.shape[:2]
        for shape in f.annotations:
            plot_shape(shape, im, color_map)
        outname = os.path.join(output_path, im_name)
        cv2.imwrite(outname, im)
