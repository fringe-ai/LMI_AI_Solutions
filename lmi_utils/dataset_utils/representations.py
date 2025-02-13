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
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnnotationType(enum.Enum):
    BOX = "Box"
    POLYGON = "Polygon"
    MASK = "Bitmask"  # This value represents a bitmask annotation.
    KEYPOINT = "Keypoint"


class Base:
    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert the dataclass to a JSON string."""
        return json.dumps(self.to_dict(), indent=4, default=self._default_serializer)

    def save(self, path: str):
        """Save the dataclass as a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    def _default_serializer(self, obj):
        """Default serializer for non-serializable objects."""
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, Base):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable")

    @classmethod
    def load(cls, path: str) -> "Base":
        """Load a dataclass instance from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)
        else:
            return cls(**data)


@dataclass
class Point2d(Base):
    x: float
    y: float

    def __init__(self, x: float, y: float):
        super().__init__()
        self.x = float(x)
        self.y = float(y)

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        rx = new_w / orig_w if orig_w else 1
        ry = new_h / orig_h if orig_h else 1
        self.x *= rx
        self.y *= ry
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
        self.x += pl
        self.y += pt
        return self

    def to_numpy(self):
        return np.array([self.x, self.y])

    def coords(self):
        return self.x, self.y

    def to_yolo(self, h, w):
        return [[self.x / w, self.y / h]]


@dataclass
class Box(Base):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    angle: float

    def __init__(self, x_min, y_min, x_max, y_max, angle):
        super().__init__()
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.angle = float(angle)
        if self.x_min > self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be less than y_max")

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("Original dimensions must be positive")
        if new_w <= 0 or new_h <= 0:
            raise ValueError("New dimensions must be positive")
        rx = new_w / orig_w
        ry = new_h / orig_h
        self.x_min *= rx
        self.x_max *= rx
        self.y_min *= ry
        self.y_max *= ry
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
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
        # If the box is rotated, use the rotated coordinates.
        if self.angle > 0:
            rotated_coords = rotate(
                self.x_min,
                self.x_max,
                w,
                h,
                self.angle,
                rot_center="up_left",
                unit="degree",
            )
            return [[pt[0] / w, pt[1] / h] for pt in rotated_coords]
        else:
            return [[self.x_min / w, self.y_min / h, width / w, height / h]]

    def to_mask(self, **kwargs):
        # Expect kwargs to include "h", "w", and optionally "mask_type"
        img_h = kwargs.get("h")
        img_w = kwargs.get("w")
        mask_type = kwargs.get("mask_type", AnnotationType.MASK)
        if mask_type == AnnotationType.MASK:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask[int(self.y_min): int(self.y_max), int(self.x_min): int(self.x_max)] = 1
            return Mask(mask=mask2rle(mask), h=img_h, w=img_w, mask_type=AnnotationType.MASK)
        elif mask_type == AnnotationType.POLYGON:
            return Polygon(
                points=[
                    Point2d(self.x_min, self.y_min),
                    Point2d(self.x_max, self.y_min),
                    Point2d(self.x_max, self.y_max),
                    Point2d(self.x_min, self.y_max),
                ]
            )
        else:
            raise ValueError("Unsupported mask_type in Box.to_mask")


class Polygon(Base):
    def __init__(self, points: List[Point2d] = None):
        super().__init__()
        self.points = points if points is not None else []

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        for point in self.points:
            point.resize(orig_h, orig_w, new_h, new_w)
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
        for point in self.points:
            point.pad(pl=pl, pt=pt)
        return self

    def to_numpy(self):
        return np.array([point.to_numpy() for point in self.points])

    def coords(self):
        return [point.coords() for point in self.points]

    def to_yolo(self, h, w):
        return [[point.x / w, point.y / h] for point in self.points]

    def to_mask(self, **kwargs):
        img_h = kwargs.get("h")
        img_w = kwargs.get("w")
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = np.array([[point.x, point.y] for point in self.points], np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return Mask(mask=mask2rle(mask), h=img_h, w=img_w, mask_type=AnnotationType.POLYGON)


@dataclass
class Mask(Base):
    mask: str

    def __init__(self, mask: Union[str, np.ndarray], h: int = 0, w: int = 0,
                 mask_type: AnnotationType = AnnotationType.MASK):
        super().__init__()
        if not isinstance(mask, (str, np.ndarray)):
            raise ValueError("Mask must be a string or numpy array")
        self.h = h
        self.w = w
        self.mask_type = mask_type
        if isinstance(mask, np.ndarray):
            self.mask = mask2rle(mask)
        else:
            self.mask = mask

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        assert orig_h > 0 and orig_w > 0, "Original height and width must be positive"
        rx = new_w / orig_w
        ry = new_h / orig_h  # fixed: use orig_h instead of orig_w here
        mask_array = rle2mask(self.mask, self.h, self.w)
        tw = int(self.w * rx)
        th = int(self.h * ry)
        resized_mask = resize(mask_array, tw, th)
        self.mask = mask2rle(resized_mask)
        self.h, self.w = th, tw
        return self

    def pad(self, **kwargs):
        pad_h = kwargs.get("pad_h", 0)
        pad_w = kwargs.get("pad_w", 0)
        mask_array = rle2mask(self.mask, self.h, self.w)
        mask_array, _, _, _, _ = fit_array_to_size(mask_array, pad_w, pad_h)
        self.mask = mask2rle(mask_array)
        self.w += pad_w
        self.h += pad_h
        return self

    def to_numpy(self):
        """Convert the mask to a numpy array."""
        return rle2mask(self.mask, self.h, self.w)

    def coords(self):
        """Return the mask coordinates as lists of x and y."""
        mask = self.to_numpy()
        ys, xs = np.nonzero(mask)
        return xs.tolist(), ys.tolist()

    def to_polygon(self) -> List[Polygon]:
        mask_array = self.to_numpy()
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.reshape(-1, 2) for contour in contours]
        return [Polygon([Point2d(x, y) for x, y in polygon]) for polygon in polygons]

    def to_yolo(self, h, w):
        # Convert mask into one or more polygon representations then into YOLO format.
        instances = []
        polygons = self.to_polygon()
        for polygon in polygons:
            instances.append(polygon.to_yolo(h, w))
        return instances

    def to_box(self, **kwargs):
        merge_boxes = kwargs.get("merge_boxes", False)
        mask_array = self.to_numpy()
        if merge_boxes:
            # Get all nonzero points and compute one bounding rectangle.
            pts = np.column_stack(np.where(mask_array > 0))
            if pts.size == 0:
                raise ValueError("Mask is empty; cannot compute bounding box.")
            x, y, w_box, h_box = cv2.boundingRect(pts)
            return Box(x_min=x, y_min=y, x_max=x + w_box, y_max=y + h_box, angle=0)
        else:
            # Compute a bounding box for each contour.
            boxes = []
            polygons = self.to_polygon()
            for poly in polygons:
                pts = np.array(poly.coords(), dtype=np.int32)
                if pts.size == 0:
                    continue
                x, y, w_box, h_box = cv2.boundingRect(pts)
                boxes.append(Box(x_min=x, y_min=y, x_max=x + w_box, y_max=y + h_box, angle=0))
            return boxes


@dataclass
class Label(Base):
    id: str
    index: str

    def __init__(self, id: str, index: Union[str, int]):
        super().__init__()
        self.id = id
        self.index = str(index) if isinstance(index, int) else index


@dataclass
class Annotation(Base):
    id: str
    label_id: str
    type: AnnotationType
    link: str = None  # Can be a Link instance
    confidence: float = 1.0
    iou: float = 0.0
    
    def __init__(
        self,
        id: str,
        label_id: str,
        type: AnnotationType,
        value: Union[Box, Mask, Point2d, Polygon],
        link: object = None,
        confidence: float = 1.0,
        iou: float = 0.0,
    ):
        super().__init__()
        self.id = id
        self.label_id = label_id
        self.type = type
        self.value = value
        self.link = link
        self.confidence = confidence
        self.iou = iou


class BoxAnnotation(Annotation):
    value: Box
    def __init__(
        self,
        id: str,
        label_id: str,
        value: Box,
        link: object = None,
        confidence: float = 1.0,
        iou: float = 0.0,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.BOX, link=link, confidence=confidence, iou=iou)
        self.value = value


class MaskAnnotation(Annotation):
    value: Mask
    
    def __init__(
        self,
        id: str,
        label_id: str,
        value: Mask,
        link: object = None,
        confidence: float = 1.0,
        iou: float = 0.0,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.MASK, link=link, confidence=confidence, iou=iou)
        self.value = value


class KeypointAnnotation(Annotation):
    value: Point2d
    def __init__(
        self,
        id: str,
        label_id: str,
        value: Point2d,
        link: object = None,
        confidence: float = 1.0,
        iou: float = 0.0,
        bounding_box_id: str = None,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.KEYPOINT, link=link, confidence=confidence, iou=iou)
        self.bounding_box_id = bounding_box_id
        self.value = value


class PolygonAnnotation(Annotation):
    value : Polygon
    def __init__(
        self,
        id: str,
        label_id: str,
        value: Polygon,
        link: object = None,
        confidence: float = 1.0,
        iou: float = 0.0,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.POLYGON, link=link, confidence=confidence, iou=iou)
        self.value = value


@dataclass
class File(Base):
    id: str
    path: str
    height: int = 0
    width: int = 0

    def __init__(self, id: str, path: str, height: int = 0, width: int = 0):
        super().__init__()
        self.id = id
        self.path = path
        self.height = height
        self.width = width


@dataclass
class FileAnnotations(Base):
    id: str  # File ID
    path: str  # File path
    height: int  # File height
    width: int  # File width
    annotations: List[Annotation]
    predictions: List[Annotation]

    def __init__(
        self,
        file: File,
        annotations: List[Annotation] = None,
        predictions: List[Annotation] = None,
    ):
        super().__init__()
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

    @property
    def has_annotations(self) -> bool:
        return len(self.annotations) > 0

    def relative_path(self, base_path: str) -> str:
        return os.path.relpath(self.path, base_path)

    def update_file(self, file: File):
        self.id = file.id
        self.path = file.path
        self.height = file.height
        self.width = file.width
        return self

    def delete_annotation(
        self, annotation_id: str, list_type: str = "annotations"
    ) -> bool:
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

    def get_annotations_by_type(
        self, annotation_type: AnnotationType, list_type: str = "annotations"
    ) -> List[Annotation]:
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        target_list = self.annotations if list_type == "annotations" else self.predictions
        return [ann for ann in target_list if ann.type == annotation_type]

    def update_annotations(
        self, annotations: List[Annotation], list_type: str = "annotations"
    ):
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        if list_type == "annotations":
            self.annotations = annotations
        else:
            self.predictions = annotations

    def assign_keypoints(self):
        def is_inside_box(point_coords, box_coords):
            x, y = point_coords
            x_min, y_min, x_max, y_max, _ = box_coords
            return x_min <= x <= x_max and y_min <= y <= y_max

        keypoints = self.get_annotations_by_type(AnnotationType.KEYPOINT)
        boxes = self.get_annotations_by_type(AnnotationType.BOX)
        for keypoint in keypoints:
            assigned = False
            for box in boxes:
                if is_inside_box(keypoint.value.coords(), box.value.coords()):
                    # Assign the box's id via the link field.
                    if keypoint.link is None:
                        keypoint.link = Link()
                    keypoint.link.annotation_id = box.id
                    assigned = True
                    break
            if not assigned:
                raise Exception(f"Keypoint {keypoint.id} not assigned to any box")
        return self


@dataclass
class Dataset(Base):
    labels: List[Label]
    files: List[FileAnnotations]

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        labels = [Label(**label_dict) for label_dict in data.get("labels", [])]
        files = []
        for file_ann_dict in data.get("files", []):
            file_obj = File(
                id=file_ann_dict["id"],
                path=file_ann_dict["path"],
                height=file_ann_dict["height"],
                width=file_ann_dict["width"],
            )
            file_ann_obj = FileAnnotations(file=file_obj)
            annotations = []
            for ann_dict in file_ann_dict.get("annotations", []):
                ann_type = AnnotationType(ann_dict["type"])
                annotation_data = ann_dict["value"]
                link_data = ann_dict.get("link", {})
                link_instance = Link(**link_data) if link_data else Link()
                if ann_type == AnnotationType.BOX:
                    box_obj = Box(**annotation_data)
                    annotation_instance = BoxAnnotation(
                        id=ann_dict["id"],
                        label_id=ann_dict["label_id"],
                        value=box_obj,
                        link=link_instance,
                        confidence=ann_dict.get("confidence", 1.0),
                        iou=ann_dict.get("iou", 0.0),
                    )
                elif ann_type == AnnotationType.MASK:
                    # Expect annotation_data to include "mask", "h", "w", and "type" (which indicates bitmask vs polygon)
                    mask_obj = Mask(
                        mask=annotation_data["mask"],
                        h=annotation_data.get("h", 0),
                        w=annotation_data.get("w", 0),
                        mask_type=AnnotationType(annotation_data["type"]),
                    )
                    annotation_instance = MaskAnnotation(
                        id=ann_dict["id"],
                        label_id=ann_dict["label_id"],
                        value=mask_obj,
                        link=link_instance,
                        confidence=ann_dict.get("confidence", 1.0),
                        iou=ann_dict.get("iou", 0.0),
                    )
                elif ann_type == AnnotationType.KEYPOINT:
                    point_obj = Point2d(**annotation_data)
                    annotation_instance = KeypointAnnotation(
                        id=ann_dict["id"],
                        label_id=ann_dict["label_id"],
                        value=point_obj,
                        link=link_instance,
                        confidence=ann_dict.get("confidence", 1.0),
                        iou=ann_dict.get("iou", 0.0),
                        bounding_box_id=ann_dict.get("bounding_box_id"),
                    )
                elif ann_type == AnnotationType.POLYGON:
                    polygon_obj = Polygon(points=[Point2d(**p) for p in annotation_data["points"]])
                    annotation_instance = PolygonAnnotation(
                        id=ann_dict["id"],
                        label_id=ann_dict["label_id"],
                        value=polygon_obj,
                        link=link_instance,
                        confidence=ann_dict.get("confidence", 1.0),
                        iou=ann_dict.get("iou", 0.0),
                    )
                else:
                    raise ValueError(f"Unsupported annotation type: {ann_type}")
                annotations.append(annotation_instance)

            file_ann_obj.annotations = annotations

            predictions = []
            for pred_dict in file_ann_dict.get("predictions", []):
                ann_type = AnnotationType(pred_dict["type"])
                annotation_data = pred_dict["value"]
                link_data = pred_dict.get("link", {})
                link_instance = Link(**link_data) if link_data else Link()
                if ann_type == AnnotationType.BOX:
                    box_obj = Box(**annotation_data)
                    annotation_instance = BoxAnnotation(
                        id=pred_dict["id"],
                        label_id=pred_dict["label_id"],
                        value=box_obj,
                        link=link_instance,
                        confidence=pred_dict.get("confidence", 1.0),
                        iou=pred_dict.get("iou", 0.0),
                    )
                elif ann_type == AnnotationType.MASK:
                    mask_obj = Mask(
                        mask=annotation_data["mask"],
                        h=annotation_data.get("h", 0),
                        w=annotation_data.get("w", 0),
                        mask_type=AnnotationType(annotation_data["type"]),
                    )
                    annotation_instance = MaskAnnotation(
                        id=pred_dict["id"],
                        label_id=pred_dict["label_id"],
                        value=mask_obj,
                        link=link_instance,
                        confidence=pred_dict.get("confidence", 1.0),
                        iou=pred_dict.get("iou", 0.0),
                    )
                elif ann_type == AnnotationType.KEYPOINT:
                    point_obj = Point2d(**annotation_data)
                    annotation_instance = KeypointAnnotation(
                        id=pred_dict["id"],
                        label_id=pred_dict["label_id"],
                        value=point_obj,
                        link=link_instance,
                        confidence=pred_dict.get("confidence", 1.0),
                        iou=pred_dict.get("iou", 0.0),
                        bounding_box_id=pred_dict.get("bounding_box_id"),
                    )
                elif ann_type == AnnotationType.POLYGON:
                    polygon_obj = Polygon(points=[Point2d(**p) for p in annotation_data["points"]])
                    annotation_instance = PolygonAnnotation(
                        id=pred_dict["id"],
                        label_id=pred_dict["label_id"],
                        value=polygon_obj,
                        link=link_instance,
                        confidence=pred_dict.get("confidence", 1.0),
                        iou=pred_dict.get("iou", 0.0),
                    )
                else:
                    raise ValueError(f"Unsupported annotation type: {ann_type}")
                predictions.append(annotation_instance)
            file_ann_obj.predictions = predictions
            files.append(file_ann_obj)
        return cls(labels=labels, files=files)

    @classmethod
    def load(cls, file_path: str) -> "Dataset":
        with open(file_path, "r") as f:
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
        raise ValueError(f"Label id {label_id} not found.")

    def to_yolo(self, **kwargs):
        to_segmentation = kwargs.get("to_segmentation", False)
        to_object_detection = kwargs.get("to_object_detection", False)
        merge_boxes = kwargs.get("merge_boxes", False)
        target_classes = kwargs.get("target_classes", ["all"])

        n_kpts = 0
        image_to_labels = {}
        base_prefix = self.base_path
        for file_ann in self.files:
            file_ann.assign_keypoints()
            file_path = file_ann.relative_path(base_prefix)
            logger.info(f"Processing file {file_path}")
            if file_path not in image_to_labels:
                image_to_labels[file_path] = []
            height = file_ann.height
            width = file_ann.width

            keypoints = file_ann.get_annotations_by_type(AnnotationType.KEYPOINT)
            if keypoints:
                if n_kpts == 0:
                    n_kpts = len(keypoints)
                elif len(keypoints) != n_kpts:
                    raise Exception(
                        f"Inconsistent number of keypoints: expected {n_kpts}, found {len(keypoints)}"
                    )

            for annotation in file_ann.annotations:
                logger.info(f"Processing annotation {annotation.id}")
                if target_classes[0] != "all" and annotation.label_id not in target_classes:
                    continue

                converted_annotations = None
                # Convert Box annotations to segmentation if requested.
                if annotation.type == AnnotationType.BOX and to_segmentation:
                    converted_value = annotation.value.to_mask(h=height, w=width, mask_type=AnnotationType.MASK)
                    annotation.value = converted_value
                    annotation.type = AnnotationType.MASK
                    logger.info(f"Converted annotation {annotation.id} from box to mask")
                # Convert MASK annotations to boxes for object detection.
                elif (annotation.type == AnnotationType.MASK and 
                      annotation.value.mask_type == AnnotationType.MASK and 
                      to_object_detection):
                    if merge_boxes:
                        box_obj = annotation.value.to_box(merge_boxes=True)
                        annotation.value = box_obj
                        annotation.type = AnnotationType.BOX
                        converted_annotations = box_obj.to_yolo(height, width)
                    else:
                        boxes = annotation.value.to_box(merge_boxes=False)
                        annotation.value = boxes
                        annotation.type = AnnotationType.BOX
                        converted_annotations = []
                        for box in boxes:
                            converted_annotations.extend(box.to_yolo(height, width))
                    logger.info(f"Converted annotation {annotation.id} from mask to box")
                elif (annotation.type == AnnotationType.MASK and 
                      annotation.value.mask_type == AnnotationType.POLYGON and 
                      to_object_detection):
                    boxes = annotation.value.to_box()  # will return a list
                    annotation.value = boxes
                    annotation.type = AnnotationType.BOX
                    converted_annotations = []
                    for box in boxes:
                        converted_annotations.extend(box.to_yolo(height, width))
                    logger.info(f"Converted annotation {annotation.id} (polygon mask) to box")
                else:
                    # For BOX and KEYPOINT (and already converted cases) call to_yolo directly.
                    if hasattr(annotation.value, "to_yolo"):
                        converted_annotations = annotation.value.to_yolo(height, width)
                    else:
                        raise ValueError(f"Annotation {annotation.id} of type {annotation.type} has no to_yolo method.")

                label_index = self.label_to_index(annotation.label_id)
                # Handle the case where converted_annotations is a list of annotations.
                if isinstance(converted_annotations, list):
                    for annot in converted_annotations:
                        instance = [label_index] + np.array(annot).flatten().tolist()
                        image_to_labels[file_path].append(instance)
                else:
                    instance = [label_index] + np.array(converted_annotations).flatten().tolist()
                    image_to_labels[file_path].append(instance)
                logger.info(f"Annotation {annotation.id} converted to YOLO format with {len(image_to_labels[file_path])} entries")
        return dict(
            image_labels=image_to_labels,
            class_map={label.index: label.id for label in self.labels},
            n_kpts=n_kpts,
        )