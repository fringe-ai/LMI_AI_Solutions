#built-in packages
import os
import logging
import cv2

#LMI packages
from label_utils import csv_utils
from label_utils.shapes import Rect, Mask, Keypoint, Brush
from system_utils.path_utils import get_relative_paths
from image_utils.img_resize import resize
from dataset_utils.representations import Dataset, File


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resize_shapes(shapes, orig_h: int, orig_w: int, new_h: int, new_w: int):
    """resize shapes in-place

    Args:
        shapes (Shape): a list of Shape objects
        rx (float): resize ratio in x direction
        ry (float): resize ratio in y direction
    """
    for annot in shapes:
        annot.value = annot.value.resize(orig_h, orig_w, new_h, new_w)
    
    return shapes

def resize_imgs_with_json(path_imgs, path_json, output_imsize, path_out, save_bg_images, recursive):
    """
    resize images and its annotations with a csv file
    if the aspect ratio changes, it will generate warnings.
    Arguments:
        path_imgs(str): the image folder
        path_csv(str): the path of csv annotation file
        output_imsize(list): a list of output image size [w,h]
    Return:
        shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    
    dataset = Dataset.load(path_json)
    for file in dataset.files:
        logger.info(f'processing {file.path}')
    base_prefix = dataset.base_path
    cnt_bg = 0
    # files = get_relative_paths(path_imgs, recursive)
    for f in dataset.files:
        file_path = f.relative_path(base_prefix)
        im_name = os.path.basename(file_path)
        im = cv2.imread(os.path.join(path_imgs, file_path))
        h,w = im.shape[:2]
        
        if not f.has_annotations:
            if not save_bg_images:
                continue
            cnt_bg += 1
            logger.info(f'{im_name}: wh of [{w},{h}] has no labels')
        else:
            logger.info(f'{im_name}: wh of [{w},{h}]')
        
        # resize image
        tw,th = output_imsize
        if tw is None and th is None:
            raise Exception('Both width and height cannot be None')
        elif tw is None:
            tw = 'w'
            rx = ry = th/h
            im2 = resize(im, height=th)
        elif th is None:
            th = 'h'
            rx = ry = tw/w
            im2 = resize(im, width=tw)
        else:
            rx,ry = tw/w, th/h
            im2 = resize(im, width=tw, height=th)
        
        out_name = os.path.splitext(im_name)[0] + f'_resized_{tw}x{th}' + '.png'
        logger.info(f'write to {out_name}')
        cv2.imwrite(os.path.join(path_out,out_name), im2)
        im2_h, im2_w = im2.shape[:2]
        f.update_file(File(path=os.path.join(path_out,out_name), width=im2_w, height=im2_h, id=f.id))
        
        # resize shapes
        shapes = resize_shapes(f.annotations,orig_h=h, orig_w=w, new_h=im2_h, new_w=im2_w)
        f.annotations = shapes
    if cnt_bg:
        logger.info(f'found {cnt_bg} images with no labels. These images will be used as background training data in YOLO')
    return dataset



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_csv', default='labels.json', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.json" in path_imgs')
    ap.add_argument('--width', type=int, default=None, help='the output image width, default=None')
    ap.add_argument('--height', type=int, default=None, help='the output image height, default=None')
    ap.add_argument('--path_out', '-o', required=True, help='the path to resized images')
    ap.add_argument('--bg', action='store_true', help='save background images that have no labels')
    ap.add_argument('--append', action='store_true', help='append to the existing output csv file')
    ap.add_argument('--recursive', action='store_true', help='search images recursively')
    args = vars(ap.parse_args())

    output_imsize = [args['width'], args['height']]
    logger.info(f'output image size: {output_imsize}')
    
    path_imgs = args['path_imgs']
    path_out = args['path_out']
    path_csv = args['path_csv'] if args['path_csv']!='labels.json' else os.path.join(path_imgs, args['path_csv'])
    
    #check if annotation exists
    if not os.path.isfile(path_csv):
        raise Exception(f'cannot find file: {path_csv}. Please create an empty csv file, if there are no labels.')
    
    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    #resize images with annotation csv file
    dataset = resize_imgs_with_json(path_imgs, path_csv, output_imsize, path_out, args['bg'], args['recursive'])
    
    dataset.save(os.path.join(path_out, 'labels.json'))

    #write csv file
    # csv_utils.write_to_csv(fname_to_shapes, os.path.join(path_out,'labels.json'), overwrite=not args['append'])
