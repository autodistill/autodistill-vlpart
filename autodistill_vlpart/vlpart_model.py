import os
import subprocess
from dataclasses import dataclass

import supervision as sv
import torch
import numpy as np
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")


def check_dependencies():
    # Create the ~/.cache/autodistill directory if it doesn't exist
    autodistill_dir = os.path.expanduser("~/.cache/autodistill")
    os.makedirs(autodistill_dir, exist_ok=True)

    os.chdir(autodistill_dir)

    # Check if CoDet is installed
    vlpart_path = os.path.join(autodistill_dir, "VLPart")

    if not os.path.isdir(vlpart_path):
        os.makedirs(vlpart_path, exist_ok=True)

        os.chdir(vlpart_path)

        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/detectron2.git"]
        )

        os.chdir("detectron2")

        subprocess.run(["pip", "install", "-e", "."])

        os.chdir(vlpart_path)

        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/VLPart.git"]
        )

        os.chdir("VLPart")

        subprocess.run(["pip", "install", "-r", "requirements.txt"])

        models_dir = os.path.join(vlpart_path, "VLPart/configs/lvis")
        os.makedirs(models_dir, exist_ok=True)

        # https://raw.githubusercontent.com/facebookresearch/VLPart/main/configs/lvis/r50_lvis.yaml
        model_config_path = os.path.join(
            vlpart_path, "VLPart/configs/lvis/r50_lvis.yaml"
        )
        subprocess.run(
            [
                "wget",
                "-O",
                model_config_path,
                "https://raw.githubusercontent.com/facebookresearch/VLPart/main/configs/lvis/r50_lvis.yaml",
            ]
        )

        # https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis.pth
        model_weights_path = os.path.join(models_dir, "r50_lvis.pth")
        subprocess.run(
            [
                "wget",
                "-O",
                model_weights_path,
                "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis.pth",
            ]
        )


check_dependencies()

import os
import sys

from .predictor import VisualizationDemo

# add VLPart to sys.path
sys.path.append(os.path.join(HOME, ".cache/autodistill/VLPart/VLPart"))

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import default_argument_parser
from vlpart.config import add_vlpart_config


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_vlpart_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.VOCABULARY = args.vocabulary
    # cfg.CUSTOM_VOCABULARY = args.custom_vocabulary
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = os.path.join(
        HOME, ".cache/autodistill/VLPart/VLPart/configs/lvis/r50_lvis.pth"
    )
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


@dataclass
class VLPart(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        # cd .cache/autodistill/VLPart/VLPart
        os.chdir(os.path.join(HOME, ".cache/autodistill/VLPart/VLPart"))

        self.ontology = ontology

        args = default_argument_parser().parse_args()

        args.config_file = os.path.join(
            HOME, ".cache/autodistill/VLPart/VLPart/configs/lvis/r50_lvis.yaml"
        )

        args.vocabulary = "lvis"

        # args.custom_vocabulary = "person,eye glasses"

        # set device
        args.device = "cpu" if not torch.cuda.is_available() else "cuda"

        args.confidence_threshold = 0.5

        cfg = setup_cfg(args)

        from detectron2.data import MetadataCatalog

        BUILDIN_METADATA_PATH = {
            "lvis": "lvis_v1_val",
        }

        lvis_thing_classes = MetadataCatalog.get(
            BUILDIN_METADATA_PATH["lvis"]
        ).thing_classes

        self.cfg = cfg
        self.demo = VisualizationDemo(cfg, args)
        self.class_names = lvis_thing_classes

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        # change to dir of this file
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        
        img = read_image(input, format="BGR")

        predictions, _ = self.demo.run_on_image(img)

        # filter predictions not in ontology
        
        selected_classes = [self.class_names.index(class_name) for class_name in self.ontology.prompts()]

        predictions = sv.Detections.from_detectron2(predictions)
        predictions = predictions[predictions.confidence > confidence]
        predictions = predictions[np.isin(predictions.class_id, selected_classes)]

        return predictions, self.class_names
