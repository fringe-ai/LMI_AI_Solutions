pip3 install -e lmi_utils/
pip3 install -e object_detectors/
pip3 install -e anomaly_detectors/
pytest tests/lmi_utils/gadget_utils/test_pipeline_utils_packaged.py
pytest tests/lmi_utils/image_utils/test_img_tile_packaged.py
pytest tests/lmi_utils/image_utils/test_tiler_packaged.py
pytest tests/object_detectors/detectron2_lmi/test_detectron2_model_packaged.py
pytest tests/object_detectors/ultralytics_lmi/yolo/test_model_yolo_packaged.py
pytest tests/object_detectors/yolov8_lmi/test_yolov8_model_packaged.py
pytest tests/anomaly_detectors/anomalib_lmi/test_anomaly_model2_packaged.py
