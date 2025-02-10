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

    def __init__(self):
        # Ensure all values are floats.
        self.x_min = float(self.x_min)
        self.y_min = float(self.y_min)
        self.x_max = float(self.x_max)
        self.y_max = float(self.y_max)
        self.angle = float(self.angle)

        # Validate that the minimum values are less than the maximum values.
        if self.x_min > self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be less than y_max")
    
    def resize_by_scale(self, rx, ry):
        # Fixed attribute names (changed from self.xmax -> self.x_max, etc.)
        self.x_max = self.x_max * rx
        self.x_min = self.x_min * rx
        self.y_max = self.y_max * ry
        self.y_min = self.y_min * ry
        return self
    
    def pad(self, pl=0, pt=0, pr=0, pb=0):
        self.x_min += pl
        self.y_min += pt
        self.x_max += pl
        self.y_max += pt
        return self
    
    def to_numpy(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])
    
        

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
        elif type == MaskType.POLYGON:
            assert isinstance(mask, list), "mask must be a list of lists"
    
    def resize_by_scale(self, rx, ry):
        if self.type == MaskType.BITMASK:
            if isinstance(self.mask, str):
                # convert the mask to a bitmask
                mask = rle2mask(self.mask, self.h, self.w)
                # Optionally, cast the new dimensions to int if needed by your resize function
                tw = int(self.w * rx)
                th = int(self.h * ry)
                # resize the mask
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
                # convert the mask to a bitmask
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
        if self.mask.type == MaskType.BITMASK:
            return rle2mask(self.mask, self.h, self.w)
        elif self.mask.type == MaskType.POLYGON:
            return np.array(self.mask)
    
    def coords(self):
        if type == MaskType.BITMASK:
            mask = self.to_numpy()
            mask = np.where(mask > 0)
            return mask[0].tolist(), mask[1].tolist()
        elif type == MaskType.POLYGON:
            mask = np.array(self.mask)
            return self.mask[:,0].tolist(), self.mask[:,1].tolist()
        

@dataclass
class Point(Base):
    x: float
    y: float
    z: float
    
    def resize_by_scale(self, rx, ry, rz=1):
        self.x = self.x * rx
        self.y = self.y * ry
        if self.z > 0:
            self.z = self.z * rz
        return self
    
    def pad(self,ph=0, pw=0, pt=0, pl=0):
        self.x += pl
        self.y += pt
        return self

@dataclass
class Label(Base):
    id: str
    name: str

@dataclass
class Link(Base):
    prediction_id: str = None
    annotation_id: str = None

@dataclass
class Annotation(Base):
    id: str
    label_id: str
    type: AnnotationType
    annotation: Union[Box, Mask, Point]
    link: Link 
    confidence: float
    iou: float = 0.0

@dataclass
class File(Base):
    id: str
    path: str
    height: int = 0
    width: int = 0

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
    
    @property
    def has_annotations(self) -> bool:
        return len(self.annotations) > 0

    # Removed the @property decorator from update_file since it takes an argument.
    def update_file(self, file: File):
        self.id = file.id
        self.path = file.path
        self.height = file.height
        self.width = file.width
        return self

    def delete_annotation(self, annotation_id: str, list_type: str = "annotations") -> bool:
        """
        Delete an annotation by its id from the specified list.

        Args:
            annotation_id (str): The id of the annotation to delete.
            list_type (str): The list from which to delete the annotation. 
                             Use "annotations" (default) or "predictions".

        Returns:
            bool: True if an annotation was deleted, False otherwise.
        """
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        
        target_list = self.annotations if list_type == "annotations" else self.predictions
        
        # Find the index of the annotation with the matching id
        for index, ann in enumerate(target_list):
            if ann.id == annotation_id:
                del target_list[index]
                logger.info(f"Deleted annotation with id '{annotation_id}' from {list_type}.")
                return True
        
        logger.warning(f"Annotation with id '{annotation_id}' not found in {list_type}.")
        return False

@dataclass
class Dataset(Base):
    labels: List[Label]
    files: List[FileAnnotations]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Dataset':
        # Parse labels
        labels = [Label(**label_dict) for label_dict in data.get('labels', [])]

        # Parse files and their annotations
        files = []
        for file_ann_dict in data.get('files', []):
            # Create File instance (assuming the file info is under key 'file')
            file_obj = File(**file_ann_dict['file'])
            annotations = []
            for ann_dict in file_ann_dict.get('annotations', []):
                # Convert annotation type from string to enum
                ann_type = AnnotationType(ann_dict['type'])
                annotation_data = ann_dict['annotation']
                # Depending on the annotation type, create the correct object
                if ann_type == AnnotationType.BOX:
                    annotation_obj = Box(**annotation_data)
                elif ann_type == AnnotationType.MASK:
                    # For Mask, also convert the mask type to MaskType
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

                # Convert the 'link_to' key into a Link instance.
                link_data = ann_dict.get('link_to', {})
                link_instance = Link(**link_data) if link_data else Link()
                
                annotations.append(Annotation(
                    id=ann_dict['id'],
                    label_id=ann_dict['label_id'],
                    type=ann_type,
                    annotation=annotation_obj,
                    link=link_instance,
                    confidence=ann_dict['confidence'],
                    iou=ann_dict.get('iou', 0.0)
                ))
            # Optionally, parse predictions if provided
            predictions = []
            for pred_dict in file_ann_dict.get('predictions', []):
                ann_type = AnnotationType(pred_dict['type'])
                annotation_data = pred_dict['annotation']
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

                link_data = pred_dict.get('link_to', {})
                link_instance = Link(**link_data) if link_data else Link()
                
                predictions.append(Annotation(
                    id=pred_dict['id'],
                    label_id=pred_dict['label_id'],
                    type=ann_type,
                    annotation=annotation_obj,
                    link=link_instance,
                    confidence=pred_dict['confidence'],
                    iou=pred_dict.get('iou', 0.0)
                ))
            files.append(FileAnnotations(
                file=file_obj,
                annotations=annotations,
                predictions=predictions
            ))
        return cls(labels=labels, files=files)

    @classmethod
    def load(cls, file_path: str) -> 'Dataset':
        """
        Load a JSON file from the given file path and return a Dataset instance.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @property
    def base_path(self) -> str:
        """
        Get the base path of the dataset.
        """
        all_files = [file_ann.path for file_ann in self.files]
        common_prefix = os.path.commonprefix(all_files)
        return os.path.dirname(common_prefix)