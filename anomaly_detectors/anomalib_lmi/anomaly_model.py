import os, sys
import logging 
from collections import OrderedDict, namedtuple
import tensorrt as trt
import torch
import numpy as np
import warnings
import cv2
import glob
import shutil
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnomalyModel')

PASS = 'PASS'
FAIL = 'FAIL'

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

class AnomalyModel:
    
    def __init__(self, engine_path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
        with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        for i in range(model.num_bindings):
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            shape = tuple(self.context.get_tensor_shape(name))
            logger.info(f'binding {name} ({dtype}) with shape {shape}')
            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)
            im = self.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def preprocess(self, image):
        img = cv2.resize(AnomalyModel.normalize(image), self.bindings['input'].shape[-2:], interpolation=cv2.INTER_AREA)
        input_dtype = np.float16 if self.fp16 else np.float32
        input_batch = np.array(np.repeat(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), 1, axis=0), dtype=input_dtype)
        return self.from_numpy(input_batch)

    def warmup(self):
        logger.info("warmup started")
        t0 = time.time()
        shape = [i*2 for i in self.bindings['input'].shape[-2:]]+[3,]
        self.predict(np.zeros(shape))
        logger.info(f"warmup ended - {time.time() - t0}")

    def predict(self, image):
        input_batch = self.preprocess(image)
        self.binding_addrs['input'] = int(input_batch.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        outputs = {x:self.bindings[x].data.cpu().numpy() for x in self.output_names}
        return outputs['output']

        
    def postprocess(self, orig_image, anomaly_map, err_thresh, err_size, mask=None):
        h,w = orig_image.shape[:2]
        anomaly_map = np.squeeze(anomaly_map.transpose((0,2,3,1)))
        if mask is not None:
            mask = cv2.resize(mask, self.bindings['input'].shape[-2:])
            anomaly_map_fp32 = anomaly_map.astype(np.float32)
            anomaly_map = cv2.bitwise_and(anomaly_map_fp32, anomaly_map_fp32, mask=mask)
        ind = anomaly_map<err_thresh
        err_count = np.count_nonzero(ind==False)
        details = {'emax':round(anomaly_map.max().tolist(), 1), 'ecnt':err_count}
        if err_count<=err_size:
            decision=PASS
            annot=orig_image
        else:
            decision=FAIL
            anomaly_map[ind] = 0
            annot = AnomalyModel.annotate(orig_image.astype(np.uint8), cv2.resize(anomaly_map.astype(np.uint8), (w, h)))
        cv2.putText(annot,
                    text=f'ad:{decision},'+ str(details).strip("{}").replace(" ","").replace("\'",""),
                    org=(4,h-20), fontFace=0, fontScale=1, color=[225, 255, 255],
                    thickness=2,
                    lineType=cv2.LINE_AA)
        return decision, annot, details

    @staticmethod
    def convert_trt(onnx_path, out_engine_path, fp16=True, workspace=4096):
        """
        convert an onnx to trt engine
        """
        logger.info(f"converting {onnx_path}...")
        assert(out_engine_path.endswith(".engine")), f"trt engine file must end with '.engine'"
        target_dir = os.path.dirname(out_engine_path)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        convert_cmd = (f'trtexec --onnx={onnx_path} --saveEngine={out_engine_path}'
                       ' --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw'
                       f' --workspace={workspace}') + (' --fp16' if fp16 else ' ')
        os.system(convert_cmd)
        os.system(f"cp {os.path.dirname(onnx_path)}/metadata.json {os.path.dirname(out_engine_path)}")

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the given mean and standard deviation for consistency with pytorch backbone
        """
        image = image.astype(np.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image /= 255.0
        image -= mean
        image /= std
        return image
    
    @staticmethod
    def normalize_anomaly_map(data, mi=None, ma=None):
        """  normalize to [0,1) """
        data_min=mi or data.min()
        data_max=ma or data.max()
        return (data - data_min) / (data_max - data_min + 1e-16)

    @staticmethod
    def annotate(img_original, heat_map_rsz):
        residual_gray = (AnomalyModel.normalize_anomaly_map(heat_map_rsz)*255).astype(np.uint8)
        residual_bgr = cv2.applyColorMap(np.expand_dims(residual_gray,-1), cv2.COLORMAP_TURBO)
        residual_rgb = cv2.cvtColor(residual_bgr, cv2.COLOR_BGR2RGB)
        annot = cv2.addWeighted(img_original, 0.5, residual_rgb, 0.5, 0)
        ind = heat_map_rsz==0
        annot[ind] = img_original[ind]
        return annot

def test(engine_dir, images_path):
    """test trt engine"""

    import time

    logger.info(f"input engine_dir is {engine_dir}")

    images = glob.glob(f"{images_path}/*.png")
    logger.info(f"{len(images)} images from {images_path}")
    if not images:
        return

    engine_path = os.path.join(engine_dir, "model.engine")
    if not os.path.isfile(engine_path):
        all = sorted([x for x in os.walk(engine_dir)][0][1])
        if all:
            engine_path = os.path.join(engine_dir, all[-1], "model.engine")
    assert(os.path.isfile(engine_path)), f"onnx file does not exist - {engine_path}"

    logger.info(f"testing engine_path {engine_path}...")

    pc = AnomalyModel(engine_path)

    out_path = f"/app/out/predictions/{model}/{os.path.basename(images_path)}"
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    pc.warmup()

    proctime = []
    for i in range(1):
        for image_path in images:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            t0 = time.time()
            decision, annotation, outputs = pc.predict(img) # 16.5859 56.07148
            proctime.append(time.time() - t0)
            logger.info(f"decision {decision},\toutputs {outputs}")
            annotation = cv2.cvtColor(annotation, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{out_path}/{os.path.basename(image_path)}", annotation.astype(np.uint8))
        
    if proctime:
        proctime = np.asarray(proctime)
        logger.info(f'Min Proc Time: {proctime.min()}')
        logger.info(f'Max Proc Time: {proctime.max()}')
        logger.info(f'Avg Proc Time: {proctime.mean()}')
        logger.info(f'Median Proc Time: {np.median(proctime)}')
    logger.info(f"Test results saved to {out_path}")

def convert(onnx_dir, fp16=True):
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    if not os.path.isfile(onnx_path):
        all = sorted([x for x in os.walk(onnx_dir)][0][1])
        if all:
            onnx_path = os.path.join(onnx_dir, all[-1], "model.onnx")
    assert(os.path.isfile(onnx_path)), f"onnx file does not exist - {onnx_path}"

    engine_out_path = f'/app/out/engines/{model}/{datetime.now().strftime("%Y-%m-%d-%H-%M")}/model.engine'
    AnomalyModel.convert_trt(onnx_path, engine_out_path, fp16=True)

if __name__ == '__main__':
    action, model, onnx_dir, engine_dir = sys.argv[1:5]
    if action in ('convert', 'all'):
        convert(onnx_dir)
    if action in ('test', 'all'):
        test(engine_dir, images_path=f'/app/data/test')