# Compile tensorrtx

CONVERSION_NAME="$(date +'%Y-%m-%d-%H-%M')"
OUTPUT_PATH=/app/trt_engines/$CONVERSION_NAME
mkdir -p $OUTPUT_PATH

OUT_PREFIX=model
WEIGHT_PATH=/app/weight.pt
CONFIG_PATH=/app/config.yaml

# generate weights
python3 /app/LMI_AI_Solutions/object_detectors/yolov5/gen_wts.py -w "$WEIGHT_PATH" -o "$OUTPUT_PATH"/"$OUT_PREFIX".wts

# build engine
/app/build_trt/yolov5 -c "$CONFIG_PATH" -w "$OUTPUT_PATH"/$OUT_PREFIX.wts -o "$OUTPUT_PATH"/model.engine
# copy production shared object
cp /app/build_trt/libmyplugins.so $OUTPUT_PATH/libmyplugins.so
