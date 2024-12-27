import cv2
import numpy as np
import torch
import os
import collections
import logging
from typing import Union
import time

from ultralytics.utils import ops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import smart_inference_mode

# import LMI AI Solutions modules
from od_base import ODBase
import gadget_utils.pipeline_utils as pipeline_utils


class ObjectDetector:
    _registry = {}

    @classmethod
    def register(cls, versions, model_names, tasks):
        """
        Decorator to register a wrapper class for multiple versions, model names, and tasks.
        """
        def decorator(wrapper_cls):
            for version in versions:
                for model_name in model_names:
                    for task in tasks:
                        key = (version, model_name, task)
                        print(f"key={key}")
                        if key in cls._registry:
                            raise ValueError(f"Wrapper already registered for version={version}, model_name='{model_name}', task='{task}'.")
                        cls._registry[key] = wrapper_cls
            return wrapper_cls
        return decorator

    def __new__(cls, version, model_name, task, *args, **kwargs):
        """
        Override __new__ to return the appropriate wrapper instance based on version, model_name, and task.
        """
        key = (version, model_name, task)
        if key not in cls._registry:
            raise ValueError(f"Unsupported combination: version={version}, model_name='{model_name}', task='{task}'.")
        wrapper_cls = cls._registry[key]
        return wrapper_cls(*args, **kwargs)

@smart_inference_mode()
def to_numpy(data):
    """Converts a tensor or a list to numpy arrays.

    Args:
        data (torch.Tensor | list): The input tensor or list of tensors.

    Returns:
        (np.ndarray): The converted numpy array.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f'Data type {type(data)} not supported')


@ObjectDetector.register(versions=[1, 2], model_names=["Yolov8"], tasks=["object_detection"])
class Yolov8(ODBase):
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, model_path:str, device='gpu', data=None, fp16=False, *args, **kwargs) -> None:
        """init the model
        Args:
            model_path (str): the path to the model_path file.
            device (str, optional): GPU or CPU device. Defaults to 'gpu'.
            data (str, optional): the path to dataset yaml file. Defaults to None.
            fp16 (bool, optional): Whether to use fp16. Defaults to False.
        Raises:
            FileNotFoundError: _description_
        """
        self.logger.info(f'Loading model from {args}')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'File not found: {model_path}')
        
        # set device
        self.device = torch.device('cpu')
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')  
            else:
                self.logger.warning('GPU not available, using CPU')
        
        # load model
        self.model = AutoBackend(model_path, self.device, data=data, fp16=fp16)
        self.model.eval()
        
        # class map < id: class name >
        self.names = self.model.names
        
        
    @smart_inference_mode()
    def forward(self, im):
        return self.model(im)
        
        
    @smart_inference_mode()
    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    @smart_inference_mode()
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
        
        
    @smart_inference_mode()
    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (np.ndarray | tensor): BCHW for tensor, [(HWC) x B] for list.
        """
        if isinstance(im, np.ndarray):
            im = self.from_numpy(im)
        
        im = im.to(self.device)
        # convert to HWC
        if im.ndim == 2:
            im = im.unsqueeze(-1)
        if im.shape[-1] ==1:
            im = im.expand(-1,-1,3)
            
        im = im.unsqueeze(0) # HWC -> BHWC
        img = im.permute((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        img = img.contiguous()

        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img
    
    
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
        return self.preprocess(im0.copy()),im0
    

    def get_min_conf(self, conf:Union[float, dict]):
        """Get the minimum confidence level for non-maximum suppression.

        Args:
            conf (float | dict): int or dictionary of <class: confidence level>.
        """
        if isinstance(conf, float):
            conf2 = conf
        elif isinstance(conf, dict):
            conf2 = 1
            class_names = set(self.model.names.values())
            for k,v in conf.items():
                if k in class_names:
                    conf2 = min(conf2, v)
            if conf2 == 1:
                self.logger.warning('No class matches in confidence dict, set to 1.0 for all classes.')
        else:
            raise TypeError(f'Confidence type {type(conf)} not supported')
        return conf2
    
    
    @smart_inference_mode()
    def postprocess(self, preds, img, orig_imgs, conf: Union[float, dict], iou=0.45, agnostic=False, max_det=300, return_segments=True):
        """Postprocesses predictions and returns a list of Results objects.
        
        Args:
            preds (torch.Tensor | list): Predictions from the model.
            img (torch.Tensor): the preprocessed image
            orig_imgs (np.ndarray | torch.Tensor | list): Original image or list of original images. If this is a tensor or a list of tensors, this function will return tensor results.
            conf (float | dict): int or dictionary of <class: confidence level>.
            iou (float): The IoU threshold below which boxes will be filtered out during NMS.
            max_det (int): The maximum number of detections to return. defaults to 300.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            return_segments(bool): If True, return the segments of the masks.
        Rreturns:
            (dict): the dictionary contains several keys: boxes, scores, classes, masks, and (masks, segments if use a segmentation model).
                    the shape of boxes is (B, N, 4), where B is the batch size and N is the number of detected objects.
                    the shape of classes and scores are both (B, N).
                    the shape of masks: (B, H, W, 3), where H and W are the height and width of the input image.
                    the shape of segments: [ (n1,2), (n2,2), ...]
        """
        
        if isinstance(preds, (list,tuple)):
            # select only inference output
            predict_mask = True if preds[0].shape[1] != 4+len(self.model.names) else False
        elif isinstance(preds, torch.Tensor):
            predict_mask = False
        else:
            raise TypeError(f'Prediction type {type(preds)} not supported')
        
        proto = None
        if predict_mask:
            proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
            preds = preds[0]
        
        conf2 = self.get_min_conf(conf)
        preds2 = ops.non_max_suppression(preds,conf2,iou,agnostic=agnostic,max_det=max_det,nc=len(self.model.names))
            
        results = collections.defaultdict(list)
        for i, pred in enumerate(preds2): # pred2: [x1, y1, x2, y2, conf, cls, mask1, mask2 ...]
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            return_tensor = isinstance(orig_img, torch.Tensor)
            
            if not len(pred):  # skip empty boxes
                continue
            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            xyxy,confs,clss = pred[:, :4],pred[:, 4],pred[:, 5]
            classes = np.array([self.model.names[c.item()] for c in clss])
            
            # filter based on conf
            if isinstance(conf, float):
                thres = np.array([conf]*len(clss))
            if isinstance(conf, dict):
                # set to 1 if c is not in conf
                thres = np.array([conf.get(c,1) for c in classes])
            M = confs > self.from_numpy(thres)
            
            if predict_mask:
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
                masks = masks[M]
                results['masks'].append(masks if return_tensor else masks.cpu().numpy())
                if return_segments:
                    segments = [ops.scale_coords(masks.shape[1:], x, orig_img.shape, normalize=False) 
                                for x in ops.masks2segments(masks)]
                    if return_tensor:
                        segments = [self.from_numpy(x) for x in segments]   # list of [ (n1,2), (n2,2), ... ]
                    results['segments'].append(segments) 
                    
            if return_tensor:
                results['boxes'].append(xyxy[M])
                results['scores'].append(confs[M])
            else:
                results['boxes'].append(xyxy[M].cpu().numpy())
                results['scores'].append(confs[M].cpu().numpy())
            results['classes'].append(classes[M.cpu().numpy()].tolist())
        return results


    @smart_inference_mode()
    def predict(self, image, configs, operators=[], iou=0.4, agnostic=False, max_det=300, return_segments=True, *args, **kwargs):
        """run yolov8 object detection inference. It runs the preprocess(), forward(), and postprocess() in sequence.
        It converts the results to the original coordinates space if the operators are provided.
        
        Args:
            model (Yolov8): the object detection model loaded memory
            image (np.ndarry | tensor): the input image
            configs (dict | float): a float or a dictionary of the confidence thresholds for each class, e.g., {'classA':0.5, 'classB':0.6}
            operators (list): a list of dictionaries of the image preprocess operators, such as {'resize':[resized_w, resized_h, orig_w, orig_h]}, {'pad':[pad_left, pad_right, pad_top, pad_bot]}
            iou (float): the iou threshold for non-maximum suppression. defaults to 0.4
            agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.
            max_det (int): The maximum number of detections to return. defaults to 300.
            return_segments(bool): If True, return the segments of the masks. Defaults to True.
        Returns:
            list of [results, time info]
            results (dict): a dictionary of the results, e.g., {
                'boxes': numpy or tensor
                'classes': a list of strings (NOT tensor)
                'scores': numpy or tensor
                'masks': numpy or tensor
                'segments': numpy or tensor
            }
            time_info (dict): a dictionary of the time info, e.g., {'preproc':0.1, 'proc':0.2, 'postproc':0.3}
        """
        time_info = {}
        
        # preprocess
        t0 = time.time()
        im = self.preprocess(image)
        time_info['preproc'] = time.time()-t0
        
        # infer
        t0 = time.time()
        pred = self.forward(im)
        time_info['proc'] = time.time()-t0
        
        # postprocess
        t0 = time.time()
        results = self.postprocess(pred,im,image,configs,iou,agnostic,max_det,return_segments)
        
        # return empty results if no detection
        results_dict = collections.defaultdict(list)
        if not len(results['boxes']):
            time_info['postproc'] = time.time()-t0
            return results_dict, time_info
        
        # only one image, get first batch
        boxes = results['boxes'][0]
        scores = results['scores'][0]
        classes = results['classes'][0]
        
        # deal with segmentation results
        if len(results['masks']):
            masks = results['masks'][0]
            masks = pipeline_utils.revert_masks_to_origin(masks, operators)
            results_dict['masks'] = masks
        if return_segments and len(results['segments']):
            segs = results['segments'][0]
            result_contours = [pipeline_utils.revert_to_origin(seg, operators) for seg in segs]
            results_dict['segments'] = result_contours
        
        # convert box to sensor space
        boxes = pipeline_utils.revert_to_origin(boxes, operators)
        results_dict['boxes'] = boxes
        results_dict['scores'] = scores
        results_dict['classes'] = classes
            
        time_info['postproc'] = time.time()-t0
        return results_dict, time_info
    
    
    @staticmethod
    @smart_inference_mode()
    def annotate_image(results, image, colormap=None, line_thickness=None, hide_label=False, hide_bbox=False):
        """annotate the object dectector results on the image. If colormap is None, it will use the random colors.

        Args:
            results (dict): the results of the object detection, e.g., {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[]}
            image (np.ndarray): the input image
            colors (list, optional): a dictionary of colormaps, e.g., {'class-A':(0,0,255), 'class-B':(0,255,0)}. Defaults to None.
            line_thickness (int, optional): the thickness of the bounding box. Defaults to None.
            hide_bbox (bool,optional): hide the bounding box
        Returns:
            np.ndarray: the annotated image
        """
        boxes = results['boxes']
        classes = results['classes']
        scores = results['scores']
        masks = results['masks']
        points = results['points']
        
        image = to_numpy(image).copy()
        if not len(boxes):
            return image
        
        # convert to numpy
        boxes = to_numpy(boxes)
        points = to_numpy(points)
        if len(masks):
            masks = to_numpy(masks)
        
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # annotate the image
        for i in range(len(boxes)):
            label = "{}: {:.2f}".format(classes[i], scores[i])
            args = {
                'label': None if hide_label else label, 
                'color': None if colormap is None else colormap[classes[i]], 
                'line_thickness':line_thickness, 
                'hide_bbox':hide_bbox
                }
            
            if boxes[i].shape == (4,2):
                pipeline_utils.plot_one_rbox(
                    boxes[i],
                    image,
                    **args
                )
            elif boxes[i].shape == (4,):
                pipeline_utils.plot_one_box(
                    boxes[i],
                    image,
                    masks[i] if len(masks) else None,
                    **args
                )
        # annotate the keypoints
        points = points.astype(int)
        for i in range(len(points)):
            for j in range(len(points[i])):
                cv2.circle(image, (points[i][j][0], points[i][j][1]), 4, (255,255,255), -1)
        return image



if __name__ == '__main__':
    
    # load the model
    image_path = "/home/rugved/projects/LMI_AI_Solutions/tests/assets/images/coco/5251661604_be7fc462e7_z.jpg"
    od_model = ObjectDetector(1, 'Yolov8', 'object_detection', model_path='/home/rugved/projects/LMI_AI_Solutions/tests/assets/models/od/yolov8n.pt', device='gpu', fp16=False)
    od_model.warmup()
    im = cv2.imread(image_path)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    results, time_info = od_model.predict(cv2.resize(rgb, (640, 640)), configs=0.5, operators=[])
    print(results)