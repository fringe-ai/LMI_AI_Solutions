from dataclasses import dataclass, asdict
import enum
import json
from typing import Union, List
from dataset_utils.mask_encoder import rle2mask, mask2rle
from image_utils.img_resize import resize
from gadget_utils.pipeline_utils import fit_array_to_size
import os
import numpy as np
import logging
from label_utils.bbox_utils import rotate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnnotationType(enum.Enum):
    BOX = 'BOX'
    MASK = 'MASK'
    KEYPOINT = 'KEYPOINT'

class MaskType(enum.Enum):
    BITMASK = 'BITMASK'
    POLYGON = 'POLYGON'

class Base:
    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert the dataclass to a JSON string."""
        return json.dumps(self.to_dict(), indent=4, default=self._default_serializer)
    
    def save(self, path: str):
        """Save the dataclass as a JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    def _default_serializer(self, obj):
        """Default serializer for non-serializable objects."""
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, Base):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable")

    def from_dict(self, data: dict) -> 'Base':
        """Create a dataclass instance from a dictionary."""
        return self.__class__(**data)
    
    def load(self, path: str) -> 'Base':
        """Load a dataclass instance from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return self.from_dict(data)

@dataclass
class Box(Base):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    angle: float

    def __init__(self, x_min, y_min, x_max, y_max, angle):
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.angle = float(angle)

        if self.x_min > self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be less than y_max")
    
    def resize(self, h=None, w=None):
        if h is None and w is None:
            raise Exception('Both height and width cannot be None')
        rx = w/self.w if w is not None else 1
        ry = h/self.h if h is not None else 1
        self.x_max = self.x_max * rx
        self.x_min = self.x_min * rx
        self.y_max = self.y_max * ry
        self.y_min = self.y_min * ry
        return self
    
    def pad(self, pad_h=0, pad_w=0, pl=0, pt=0):
        self.x_min += pl
        self.y_min += pt
        self.x_max += pl
        self.y_max += pt
        return self
    
    def to_numpy(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max, self.angle])
    
    def coords(self):
        return self.x_min, self.y_min, self.x_max, self.y_max, self.angle
    
    def to_yolo(self, h, w):
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        if self.angle > 0:
            rotated_coords = rotate(self.x_min, self.x_max, w, h, self.angle, rot_center='up_left', unit='degree')
            xyxy = [
               [pt[0] / w, pt[1] / h] for pt in rotated_coords
            ]
            return xyxy
        else:
            return [[self.x_min / w, self.y_min / h, width / w, height / h]]
    
    def to_mask(self, **kwargs):
        if kwargs.get('mask_type') == MaskType.BITMASK:
            mask = np.zeros((kwargs.get('h'), kwargs.get('w')), dtype=np.uint8)
            mask[int(self.y_min):int(self.y_max), int(self.x_min):int(self.x_max)] = 1
            return Mask(
                mask=mask2rle(mask),
                type=MaskType.BITMASK,
                h=kwargs.get('h'),
                w=kwargs.get('w'),
            )
        else:
            return Mask(
                mask=[[self.x_min, self.y_min], [self.x_max, self.y_min], [self.x_max, self.y_max], [self.x_min, self.y_max]],
                type=MaskType.POLYGON
            )
          

@dataclass
class Mask(Base):
    type: MaskType
    mask: Union[str, List[List[float]]]
    h: int = 0
    w: int = 0

    def __init__(self, type: MaskType, mask: Union[str, List[List[float]]], h: int = 0, w: int = 0):
        self.type = type
        self.mask = mask
        self.h = h
        self.w = w
        if type == MaskType.BITMASK:
            assert self.h > 0 and self.w > 0, "height and width must be provided for BITMASK type"
            if isinstance(mask, np.ndarray):
                assert mask.shape == (self.h, self.w), "mask shape must match height and width"
                self.mask = mask2rle(mask)
        elif type == MaskType.POLYGON:
            assert isinstance(mask, list), "mask must be a list of lists"

    
    def resize(self, h=None, w=None):
        if h is None and w is None:
            raise Exception('Both height and width cannot be None')
        rx = w/self.w if w is not None else 1
        ry = h/self.h if h is not None else 1
            
        if self.type == MaskType.BITMASK:
            if isinstance(self.mask, str):
                mask = rle2mask(self.mask, self.h, self.w)
                tw = int(self.w * rx)
                th = int(self.h * ry)
                self.mask = mask2rle(resize(mask, tw, th))
                self.h = th
                self.w = tw
                
        elif self.type == MaskType.POLYGON:
            mask = np.array(self.mask)
            mask[:,0] = mask[:,0] * rx
            mask[:,1] = mask[:,1] * ry
            self.mask = mask.tolist()
        return self
    
    def pad(self, ph=0, pw=0, pt=0, pl=0):
        if self.type == MaskType.BITMASK:
            if isinstance(self.mask, str):
                mask = rle2mask(self.mask, self.h, self.w)
                mask, _, _, _, _ = fit_array_to_size(mask, ph, pw)
                self.mask = mask2rle(mask)
                self.h = ph
                self.w = pw
                
        elif self.type == MaskType.POLYGON:
            mask = np.array(self.mask)
            mask[:,0] += pl
            mask[:,1] += pt
            self.mask = mask.tolist()
        return self

    def to_numpy(self):
        if self.type == MaskType.BITMASK:
            return rle2mask(self.mask, self.h, self.w)
        elif self.type == MaskType.POLYGON:
            return np.array(self.mask)
    
    def coords(self):
        if self.type == MaskType.BITMASK:
            mask = self.to_numpy()
            mask = np.where(mask > 0)
            return mask[0].tolist(), mask[1].tolist()
        elif self.type == MaskType.POLYGON:
            mask = np.array(self.mask)
            return mask[:,0].tolist(), mask[:,1].tolist()
        

@dataclass
class Point(Base):
    x: float
    y: float
    z: float
    
    def resize(self, h=None, w=None, z=None):
        if h is None and w is None:
            raise Exception('Both height and width cannot be None')
        rx = w/self.w if w is not None else 1
        ry = h/self.h if h is not None else 1
        rz = z/self.z if z is not None else 1
        self.x = self.x * rx
        self.y = self.y * ry
        if self.z > 0:
            self.z = self.z * rz
        return self
    
    def pad(self, ph=0, pw=0, pt=0, pl=0):
        self.x += pl
        self.y += pt
        return self
    
    def to_numpy(self):
        return np.array([self.x, self.y, self.z])
    
    def coords(self):
        return self.x, self.y, self.z
    
    def to_yolo(self, h, w):
        return [[self.x / w, self.y / h]]
        

@dataclass
class Label(Base):
    id: str
    index: str
    
    def __init__(self, id: str, index: Union[str, int]):
        self.id = id
        self.index = str(index) if isinstance(index, int) else index

@dataclass
class Link(Base):
    prediction_id: str = None
    annotation_id: str = None

@dataclass
class Annotation(Base):
    id: str
    label_id: str
    type: AnnotationType
    value: Union[Box, Mask, Point]
    link: Link 
    confidence: float
    iou: float = 0.0
    

@dataclass
class File(Base):
    id: str
    path: str
    height: int = 0
    width: int = 0

@dataclass
class FileAnnotations(Base):
    id: str  # File ID
    path: str  # File path
    height: int  # File height
    width: int  # File width
    annotations: List[Annotation]
    predictions: List[Annotation]
    
    def __init__(self, file: File, annotations: List[Annotation] = None, predictions: List[Annotation] = None):
        if annotations is None:
            annotations = []
        if predictions is None:
            predictions = []
        self.id = file.id
        self.path = file.path
        self.height = file.height
        self.width = file.width
        self.annotations = annotations
        self.predictions = predictions
    
    def relative_path(self, base_path: str) -> str:
        return os.path.relpath(self.path, base_path)
    
    def get_filename(self) -> str:
        return os.path.basename(self.path)
    
    @property
    def has_annotations(self) -> bool:
        return len(self.annotations) > 0

    def update_file(self, file: File):
        self.id = file.id
        self.path = file.path
        self.height = file.height
        self.width = file.width
        return self

    def delete_annotation(self, annotation_id: str, list_type: str = "annotations") -> bool:
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        
        target_list = self.annotations if list_type == "annotations" else self.predictions
        
        for index, ann in enumerate(target_list):
            if ann.id == annotation_id:
                del target_list[index]
                logger.info(f"Deleted annotation with id '{annotation_id}' from {list_type}.")
                return True
        
        logger.warning(f"Annotation with id '{annotation_id}' not found in {list_type}.")
        return False
    
    def get_annotations_by_type(self, annotation_type: AnnotationType, list_type: str = "annotations") -> List[Annotation]:
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        
        target_list = self.annotations if list_type == "annotations" else self.predictions
        return [ann for ann in target_list if ann.type == annotation_type]
    
    def update_annotations(self, annotations: List[Annotation], list_type: str = "annotations"):
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        
        if list_type == "annotations":
            self.annotations = annotations
        else:
            self.predictions = annotations
    
    
    def assign_keypoints(self):
        
        def is_inside_box(point: Point, box: Box):
            x, y = point
            x_min, y_min, x_max, y_max, angle = box.coords()
            return x_min <= x <= x_max and y_min <= y <= y_max
        
        boxes = self.get_annotations_by_type(AnnotationType.BOX)
        for annot in self.annotations:
            if annot.type == AnnotationType.KEYPOINT:
                keypoint = annot
            else:
                continue
            assigned = False
            for box in boxes:
                if is_inside_box(keypoint.value.coords(), box.value.coords()):
                    keypoint.link.annotation_id = box.id
                    assigned = True
                    break
            if not assigned:
                raise Exception('Key point not assigned to any box')
        return self
    
@dataclass
class Dataset(Base):
    labels: List[Label]
    files: List[FileAnnotations]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Dataset':
        labels = [Label(**label_dict) for label_dict in data.get('labels', [])]
        files = []
        for file_ann_dict in data.get('files', []):
            # Create File instance
            annotations = []
            for ann_dict in file_ann_dict.get('annotations', []):
                ann_type = AnnotationType(ann_dict['type'])
                annotation_data = ann_dict['value']
                if ann_type == AnnotationType.BOX:
                    annotation_obj = Box(**annotation_data)
                elif ann_type == AnnotationType.MASK:
                    annotation_obj = Mask(
                        type=MaskType(annotation_data['type']),
                        mask=annotation_data['mask'],
                        h=annotation_data.get('h', 0),
                        w=annotation_data.get('w', 0)
                    )
                elif ann_type == AnnotationType.KEYPOINT:
                    annotation_obj = Point(**annotation_data)
                else:
                    raise ValueError(f"Unsupported annotation type: {ann_type}")

                link_data = ann_dict.get('link', {})
                link_instance = Link(**link_data) if link_data else Link()
                
                annotations.append(Annotation(
                    id=ann_dict['id'],
                    label_id=ann_dict['label_id'],
                    type=ann_type,
                    value=annotation_obj,
                    link=link_instance,
                    confidence=ann_dict['confidence'],
                    iou=ann_dict.get('iou', 0.0)
                ))
            predictions = []
            for pred_dict in file_ann_dict.get('predictions', []):
                ann_type = AnnotationType(pred_dict['type'])
                annotation_data = pred_dict['value']
                if ann_type == AnnotationType.BOX:
                    annotation_obj = Box(**annotation_data)
                elif ann_type == AnnotationType.MASK:
                    annotation_obj = Mask(
                        type=MaskType(annotation_data['type']),
                        mask=annotation_data['mask'],
                        h=annotation_data.get('h', 0),
                        w=annotation_data.get('w', 0)
                    )
                elif ann_type == AnnotationType.KEYPOINT:
                    annotation_obj = Point(**annotation_data)
                else:
                    raise ValueError(f"Unsupported annotation type: {ann_type}")

                link_data = pred_dict.get('link', {})
                link_instance = Link(**link_data) if link_data else Link()
                
                predictions.append(Annotation(
                    id=pred_dict['id'],
                    label_id=pred_dict['label_id'],
                    type=ann_type,
                    value=annotation_obj,
                    link=link_instance,
                    confidence=pred_dict['confidence'],
                    iou=pred_dict.get('iou', 0.0)
                ))
            file_ann_obj = FileAnnotations(
                file=File(id=file_ann_dict['id'], path=file_ann_dict['path'], height=file_ann_dict['height'], width=file_ann_dict['width']),
                annotations=annotations
            )
            files.append(file_ann_obj)
        return cls(labels=labels, files=files)

    @classmethod
    def load(cls, file_path: str) -> 'Dataset':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @property
    def base_path(self) -> str:
        all_files = [file_ann.path for file_ann in self.files]
        common_prefix = os.path.commonprefix(all_files)
        return os.path.dirname(common_prefix)
    
    def get_labels(self) -> List[str]:
        return [label.id for label in self.labels]
    
    def label_to_index(self, label_id: str):
        for label in self.labels:
            if label.id == label_id:
                return int(label.index)
    
    def to_yolo(self, **kwargs):
        to_segmentation = kwargs.get('to_segmentation', False)
        to_object_detection = kwargs.get('to_object_detection', False)
        merge_boxes = kwargs.get('merge_boxes', False)
        target_classes = kwargs.get('target_classes', ['all'])
        
        n_kpts = 0
        image_to_labels = {}
        base_prefix = self.base_path
        for file in self.files:
            file.assign_keypoints()
            file_path = file.relative_path(base_prefix)
            logger.info(f'Processing file {file_path}')
            if file_path not in image_to_labels:
                image_to_labels[file_path] = []
            
            height = file.height
            width = file.width
            
            keypoints = file.get_annotations_by_type(AnnotationType.KEYPOINT)
            if len(keypoints) > 0:
                if n_kpts == 0:
                    n_kpts = len(keypoints)
                elif len(keypoints) != n_kpts:
                    raise Exception(f'Inconsistent number of keypoints expected {n_kpts} found {len(keypoints)}')
                
            for annotation in file.annotations:
                logger.info(f'Processing annotation {annotation.id}')
                if target_classes[0] != 'all' and annotation.label_id not in target_classes:
                    continue
                
                if annotation.type == AnnotationType.BOX and to_segmentation:
                    converted_value = annotation.value.to_mask()
                    annotation.value = converted_value
                    annotation.type = AnnotationType.MASK
                    logger.info(f'Converted annotation {annotation.id} of type {annotation.type} to mask')
                
                elif annotation.type == AnnotationType.MASK and annotation.value.type == MaskType.BITMASK and to_object_detection:
                    if not merge_boxes:
                        converted_value = annotation.value.to_box()
                        annotation.value = converted_value
                        annotation.type = AnnotationType.BOX
                    else:
                        for annot in annotation.value.to_box(merge_boxes=True):
                            instance = [self.label_to_index(annotation.label_id)] + np.array(annot.to_yolo(height, width)).flatten().tolist()
                            image_to_labels[file_path].append(instance)
                    logger.info(f'Converted annotation {annotation.id} of type {annotation.type} to box')
                
                elif annotation.type == AnnotationType.MASK and annotation.value.type == MaskType.POLYGON and to_object_detection:
                    converted_value = annotation.value.to_box()
                    annotation.value = converted_value
                    annotation.type = AnnotationType.BOX
                    logger.info(f'Converted annotation {annotation.id} of type {annotation.type} to box')
                
                converted_annotations = annotation.value.to_yolo(height, width)
                logger.info(f'Converted annotations {len(converted_annotations)} to YOLO format')
                label_index = self.label_to_index(annotation.label_id)
                
                for annot in converted_annotations:
                    instance = [label_index] + np.array(annot).flatten().tolist()
                    image_to_labels[file_path].append(instance)


        return dict(
            image_labels = image_to_labels,
            class_map = {
                label.index : label.id for label in self.labels
            },
            n_kpts = n_kpts
        )
