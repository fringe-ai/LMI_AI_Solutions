#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
import cv2

logger = setup_logger()

def setup_cfg(config_file, opts=None):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    add_pointrend_config(cfg)
    cfg.merge_from_file(config_file)
    if opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

# experimental. API not yet final
def export_scripting(torch_model, output_dir):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
    

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(os.path.join(output_dir, "model.pt"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, output_dir)
    # TODO inference in Python now missing postprocessing glue code
    return None


# experimental. API not yet final
def export_tracing(torch_model, inputs, output_dir, format="torchscript"):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    ts_model = torch.jit.trace(traceable_model, (image,))
    with PathManager.open(os.path.join(output_dir, "model.pt"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, output_dir)
    with PathManager.open(os.path.join(output_dir, "det2onnx.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f, opset_version=STABLE_ONNX_OPSET_VERSION)
    
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


def get_sample_inputs(cfg, use_train: bool = False, sample_image: str = None):

    if sample_image is None:
        # get a first batch from dataset
        if use_train:
            data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0])
        else:
            data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = cv2.imread(sample_image)
        print(f"Processing image {sample_image}")
        print(f"Image size (h,w): {original_image.shape[:2]}")
        aug = T.ResizeShortestEdge(
            [original_image.shape[0], original_image.shape[0]], original_image.shape[0]
        )
        image = aug.get_transform(original_image).apply_image(original_image)
        height, width = original_image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


def main() -> None:
    global logger, cfg, args
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="onnx",
    )
    parser.add_argument(
        "--export-method",
        choices=["caffe2_tracing", "tracing", "scripting"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file", default='/home/weights/config.yaml')
    parser.add_argument("--sample-image", type=str, help="sample image for input", default='/home/weights/sample_image.png')
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model", default='/home/weights/onnx')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(args.config_file, args.opts)

    # create a torch model
    torch_model = build_model(cfg)
    cfg.merge_from_list(["MODEL.WEIGHTS", '/home/weights/model_final.pth'])
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    # convert and save model
    if args.export_method == "scripting":
        exported_model = export_scripting(torch_model, get_sample_inputs(cfg, use_train=False, sample_image=args.sample_image))
    elif args.export_method == "tracing":
        exported_model = export_tracing(torch_model, get_sample_inputs(cfg, use_train=False, sample_image=args.sample_image), args.format)

    logger.info("Success.")


if __name__ == "__main__":
    main()  # pragma: no cover