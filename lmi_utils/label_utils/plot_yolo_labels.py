import cv2
import argparse
import glob
import os
import numpy as np
from label_utils.plot_utils import plot_one_polygon

def read_yolo_labels(file_path, h, w, num_keypoints=None):
    """
    Reads a YOLO formatted label text file and returns a dictionary keyed by class.
    
    Supported annotation types:
      - Bounding Box (bbox): 4 values (center_x, center_y, width, height)
      - Oriented Bounding Box (obb): 8 values representing four (x, y) corner pairs:
          x1 y1 x2 y2 x3 y3 x4 y4
      - Polygon: An even number of values greater than the above formats is treated as a polygon.
      - With Keypoints (if num_keypoints is provided):
          - bbox with keypoints: 4 + 2*num_keypoints values
          - obb with keypoints: 8 + 2*num_keypoints values
          
    If a line does not match one of these expected formats, it is skipped.
    """
    labels_dict = {}

    # Read the file content and split by newline.
    with open(file_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        tokens = line.split()
        cls = tokens[0]
        try:
            # Convert remaining tokens to float values.
            values = list(map(float, tokens[1:]))
        except ValueError:
            # Skip lines with conversion issues.
            continue

        ann = {}

        if num_keypoints is not None:
            expected_bbox_kp = 4 + 2 * num_keypoints
            expected_obb_kp = 8 + 2 * num_keypoints
            if len(values) == expected_bbox_kp:
                # Bounding box with keypoints.
                ann["type"] = "bbox_with_keypoints"
                ann["x_center"] = values[0] * w
                ann["y_center"] = values[1] * h
                ann["width"] = values[2] * w
                ann["height"] = values[3] * h
                keypoints = []
                for i in range(4, len(values), 2):
                    keypoints.append((values[i] * w, values[i+1] * h))
                ann["keypoints"] = keypoints
            elif len(values) == expected_obb_kp:
                # Oriented bounding box with keypoints.
                ann["type"] = "obb_with_keypoints"
                # Extract the 8 values representing the four corners.
                ann["corners"] = [
                    (values[0] * w, values[1] * h),
                    (values[2] * w, values[3] * h),
                    (values[4] * w, values[5] * h),
                    (values[6] * w, values[7] * h)
                ]
                keypoints = []
                for i in range(8, len(values), 2):
                    keypoints.append((values[i] * w, values[i+1] * h))
                ann["keypoints"] = keypoints
            elif len(values) % 2 == 0 and len(values) >= 8:
                # Fallback: treat as polygon.
                ann["type"] = "polygon"
                points = []
                for i in range(0, len(values), 2):
                    points.append((values[i] * w, values[i+1] * h))
                ann["points"] = points
            else:
                # Unrecognized format.
                continue
        else:
            # Without keypoints.
            if len(values) == 4:
                # Standard bounding box.
                ann["type"] = "bbox"
                ann["x_center"] = values[0] * w
                ann["y_center"] = values[1] * h
                ann["width"] = values[2] * w
                ann["height"] = values[3] * h
            elif len(values) == 8:
                # Oriented bounding box with 8 numbers (four (x, y) pairs).
                ann["type"] = "obb"
                ann["corners"] = [
                    (values[0] * w, values[1] * h),
                    (values[2] * w, values[3] * h),
                    (values[4] * w, values[5] * h),
                    (values[6] * w, values[7] * h)
                ]
            elif len(values) % 2 == 0 and len(values) > 8:
                # Polygon: even number of coordinates (more than 8).
                ann["type"] = "polygon"
                points = []
                for i in range(0, len(values), 2):
                    points.append((values[i] * w, values[i+1] * h))
                ann["points"] = points
            else:
                # Unrecognized format.
                continue

        # Add the annotation under its class.
        if cls not in labels_dict:
            labels_dict[cls] = []
        labels_dict[cls].append(ann)

    return labels_dict
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", type=str, help="path to the root folder contains images and labels")
    parser.add_argument("--path_out", type=str, help="path to the output directory")
    print(f'args: {parser.parse_args()}')
    args = parser.parse_args()
    txt_files = glob.glob(os.path.join(args.path_dataset, "labels/*.txt"))
    print(f"Found {len(txt_files)} label files.")
    os.makedirs(args.path_out, exist_ok=True)
    for txt_file in txt_files:
        print(f"Reading labels from {txt_file}")
        image_path = txt_file.replace(".txt", ".png")
        image_path = image_path.replace("labels", "images")
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        labels = read_yolo_labels(txt_file, h, w)
        for label in labels:
            for i, ann in enumerate(labels[label]):
                if ann["type"] == "polygon":
                    points = np.array(ann["points"], dtype=np.int32)
                    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                elif ann["type"] == "bbox":
                    x,y,w,h = ann["x_center"], ann["y_center"], ann["width"], ann["height"]
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                elif ann["type"] == "obb":
                    corners = np.array(ann["corners"], dtype=np.int32)
                    cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
        cv2.imwrite(os.path.join(args.path_out, f"{os.path.basename(image_path)}"), image)
        