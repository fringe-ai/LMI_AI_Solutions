import cv2
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from enum import Enum
from typing import Union
from PIL import Image

from ultralytics.utils import ops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.augment import classify_transforms


class Yolov8_cls:
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, weights:str, device='gpu', data=None, crop_fraction=1) -> None:
        """init the model

        Args:
            weights (str): the path to the weights file.
            device (str, optional): _description_. Defaults to 'gpu'.
            data (str, optional): _description_. Defaults to None.

        Raises:
            FileNotFoundError: _description_
        """
        if not os.path.isfile(weights):
            raise FileNotFoundError(f'File not found: {weights}')
        
        # set device
        self.device = torch.device('cpu')
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')  
            else:
                self.logger.warning('GPU not available, using CPU')
        
        # load model
        self.model = AutoBackend(weights, self.device, data=data)
        self.model.eval()
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.model.imgsz[0], crop_fraction=crop_fraction),
            )
        )
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"
        
        
    def forward(self, im):
        return self.model(im)
        
        
    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    def warmup(self, imgsz=[640, 640]):
        """
        Warm up the model by running one forward pass with a dummy input.
        Args:
            imgsz(list): list of [h,w], default to [640,640]
        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        if isinstance(imgsz, tuple):
            imgsz = list(imgsz)
            
        imgsz = [1,3]+imgsz
        im = torch.empty(*imgsz, dtype=torch.half if self.model.fp16 else torch.float, device=self.device)  # input
        self.forward(im)  # warmup
        
        
    def preprocess(self, img):
        """Prepares input image before inference.
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/classify/predict.py#L36
        """
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(im)) for im in img], dim=0 # not convert from BGR to RGB
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
    
    
    def load_with_preprocess(self, im_path:str):
        """load image and do im preprocess

        Args:
            im_path (str): the path to the image, could be either .npy, .png, or other image formats
            
        Returns:
            (torch.Tensor): the preprocessed image.
            (np.ndarray): the original image.
        """
        ext = os.path.splitext(im_path)[-1]
        if ext=='.npy':
            im0 = np.load(im_path)
        else:
            im0 = cv2.imread(im_path) #BGR format
            im0 = im0[:,:,::-1] #BGR to RGB
        return self.preprocess(im0),im0
    
    
    def postprocess(self, preds):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            
        """
         
        results = defaultdict(list)
        for pred in preds:
            names = self.model.names
            
            self.logger.info(f'pred: {pred}, shape: {pred.shape}')
            self.logger.info(f'names: {names}')
            
            results['scores'].append( pred.cpu().numpy())
            results['classes'].append(names)
            
        return results
