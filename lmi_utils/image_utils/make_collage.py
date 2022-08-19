#%%
import glob
import cv2
import os
import numpy as np
from image_utils.img_resize import resize
from image_utils.pad_image import fit_array_to_size
import logging

logging.basicConfig(level=logging.INFO)
#%%
def gen_collage(input_path,output_path,colmax,width):

    files=glob.glob(os.path.join(input_path,'*.png'))
    n=len(files)
    files=sorted(files)
    logging.debug(f'List of files: {files}')


    nrows=n//colmax
    remainder=n%colmax

    #%%
    col_idx=np.arange(colmax)
    nrows = nrows if remainder==0 else nrows+1

    newRow=False
    currentRow=None
    lastRow=None
    collage=None

    img_h=[]
    img_w=[]
    imgs=[]
    for file in files:
        img=cv2.imread(file)
        img_h.append(img.shape[0])
        img_w.append(img.shape[1])
        imgs.append(img)
    img_h=np.array(img_h)
    img_w=np.array(img_w)
    max_h=img_h.max()
    print(f'[INFO] Max Input Image Height: {max_h}')
    max_w=img_w.max()
    print(f'[INFO] Max Input Image Width: {max_w}')

    for img in imgs:
        # img,_,_=fit_array_to_size(img,max_w,max_h)
        h,w=img.shape[:2]
        pad_h = 0 if h==max_h else max_h-h
        pad_w = 0 if w==max_w else max_w-w
        img=cv2.copyMakeBorder(img,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,None,(0,0,0))
        if width is not None:
            img=resize(img, width=width)
        if currentRow is None:    
            currentRow=img
        else:
            currentRow=np.hstack([currentRow,img])
        if col_idx[0]==colmax-1:
            lastRow=currentRow
            if collage is None:
                collage=currentRow
                currentRow=None
            else:
                collage=np.vstack([collage,currentRow])
                currentRow=None
        col_idx=np.roll(col_idx,-1)

    if (remainder != 0) and (lastRow is not None):
        padR=lastRow.shape[1]-currentRow.shape[1]
        currentRow=cv2.copyMakeBorder(currentRow,0,0,0,padR,cv2.BORDER_CONSTANT,value=(0,0,0))
        collage=np.vstack([collage,currentRow])

    cv2.imwrite(output_path,collage)

# %%
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input_data_path',required=True)
    ap.add_argument('-o','--output_image_path',required=True)
    ap.add_argument('--width',default=None)
    ap.add_argument('--max_columns',type=int,default=10)
    
    args=vars(ap.parse_args())
    input_data_path=args['input_data_path']
    output_image_path=args['output_image_path']
    max_cols=args['max_columns']
    width=args['width']
    if width is not None:
        width=int(width)


    output_path,output_file=os.path.split(output_image_path)
    output_file_ext=os.path.splitext(output_file)[1]
    if output_file_ext != '.png':
        raise Exception(f'Please change output image path from {output_file_ext} to .png')

    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)

    gen_collage(input_data_path,output_image_path,max_cols,width)
    