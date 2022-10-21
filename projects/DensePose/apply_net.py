#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import logging
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List
import torch
import csv

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

import cv2
import time
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector
import keyboard
from mediapipe.framework.formats import landmark_pb2
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

flag =1
s=time.time()
detector = FaceMeshDetector(maxFaces=1)

#Results Logging
f = open(r"DenseposeResults.csv","w")
writer = csv.writer(f)



vid = cv2.VideoCapture(1) # Change from 0 to 1 if you see an orange screen, this is caused by multiple cameras being connected
DOC = """Apply Net - a tool to print / visualize DensePose results
"""

LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, "Action"] = {}


class Action(object):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument("cfg", metavar="<config>", help="Config file")
        parser.add_argument("model", metavar="<model>", help="Model file")
        parser.add_argument("input", metavar="<input>", help="Input data")
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )



    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        logger.info(f"Loading config from {args.cfg}")
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        logger.info(f"Loading model from {args.model}")
        predictor = DefaultPredictor(cfg)
        logger.info(f"Loading data from {args.input}")
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
        context = cls.create_context(args, cfg)


        #for file_name in file_list: #----------------------------------------------------------------------------------------------------------
        flag = 1
        start = time.time()
        end=time.time()
        s = time.time()
        while 1:
            if (end - s > 1):
                if keyboard.is_pressed("space"):
                    flag = flag * -1
                    s=time.time()
                    #print(flag)
            start  = time.time()
            file_name = file_list[0]
            #img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            #img = cv2.imread('projects/DensePose/image.jpg')
            ret, frame = vid.read()
            img = frame
            """cv2.imshow('test',img)
            cv2.waitKey(1)"""
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs,flag)
            end = time.time()
            #print(end-start,'seconds, FPS:',(1/(end-start)))
            writer.writerow([end-start,1/(end-start)])
        cls.postexecute(context)

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list


@register_action
class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results.pkl",
            help="File name to save dump to",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        #ogger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            pickle.dump(context["results"], hFile)
            #logger.info(f"Output saved to {out_fname}")


@register_action
class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--texture_atlas",
            metavar="<texture_atlas>",
            default=None,
            help="Texture atlas file (for IUV texture transfer)",
        )
        parser.add_argument(
            "--texture_atlases_map",
            metavar="<texture_atlases_map>",
            default=None,
            help="JSON string of a dict containing texture atlas files for each mesh",
        )
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default="outputres.png",
            help="File name to save output to",
        )

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def sizeEstimation(cls: type,img, flag):
        if (flag < 0):
            pass
        else:
            imgface, faces = detector.findFaceMesh(img, draw=False)

            if faces:
                face = faces[0]
                pointLeft = face[145]  # Left Irisa
                pointRight = face[374]  # Right Iris

                wi, _ = detector.findDistance(pointLeft, pointRight)
                W = 6.3  # 6.3cm distance between Iris's on average
                f = 840  # focal length of camera
                d = (W) / wi

                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(imgRGB)
                if results.pose_landmarks:
                    # mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS) # shows points on body
                    # mpDraw.draw_landmarks(img,results.pose_landmarks,frozenset([(11, 12), (11, 23), (12, 24),(23, 24)]))
                    landmark_subset = landmark_pb2.NormalizedLandmarkList(
                        landmark=[
                            results.pose_landmarks.landmark[11],
                            results.pose_landmarks.landmark[12],
                            results.pose_landmarks.landmark[23],
                            results.pose_landmarks.landmark[24],
                        ]
                    )
                    mpDraw.draw_landmarks(img, landmark_subset, frozenset([(0, 1), (0, 2), (1, 3), (2, 3)]),mpDraw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                    mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w, c = img.shape
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), int(
                            lm.z * w)  # cx ,cy are pixel locations of each point
                        if (id == 11):
                            # print(id, lm)
                            cx11 = cx
                            cy11 = cy
                            cz11 = cz
                        if (id == 12):
                            cx12 = cx
                            cy12 = cy
                            cz12 = cz
                        if (id == 23):
                            cx23 = cx
                            cy23 = cy
                            cz23 = cz
                        if (id == 24):
                            cx24 = cx
                            cy24 = cy
                            cz24 = cz

                p1 = np.array([cx11, cy11])
                p2 = np.array([cx12, cy12])
                squared_dist1 = np.sum((p1 - p2) ** 2, axis=0)
                dist1 = np.sqrt(squared_dist1)

                p1 = np.array([cx12, cy12])
                p2 = np.array([cx24, cy24])
                squared_dist1 = np.sum((p1 - p2) ** 2, axis=0)
                dist2 = np.sqrt(squared_dist1)
                # shoulder,_ = w, detector.findDistance([cx11,cy11],[cx12,cy12])

                font = cv2.FONT_HERSHEY_SIMPLEX
                text1 = str(int(dist1 * d)) + "cm"
                text2 = str(int(dist2 * d)) + "cm"

                # get boundary of this text
                textsize = cv2.getTextSize(text1, font, 1, 2)[0]

                # get coords based on boundary
                # textX = int((cx11 - textsize[0]) / 2)
                # textY = int((cy11 + textsize[1]) / 2)
                textX1 = int((cx12 + ((cx11 - cx12) / 2)) - (textsize[0] / 2))
                textY1 = int(cy12) - 20

                textX2 = int(cx12) - textsize[0] - 20
                textY2 = int(cy12 + ((cy24 - cy12) / 2) - (textsize[1] / 2))

                # add text centered on image
                cv2.putText(img, text1, (textX1, textY1), font, 1, (0, 0, 0), 2)
                cv2.putText(img, text2, (textX2, textY2), font, 1, (0, 0, 0), 2)
        return img, flag, s
    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances,flag
    ):
        import cv2
        import numpy as np

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        #logger.info(f"Processing {image_fpath}")

        im = entry["image"]


        cv2.imshow('original', im)


        #im = np.tile(im[:, :, np.newaxis], [1, 1, 3])
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        #image_vis = visualizer.visualize(image, data)
        image_vis = visualizer.visualize(im, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        #cv2.imwrite(out_fname, image_vis)
        #image_vis = cv2.cvtColor(image_vis,cv2.COLOR_GRAY2RGB)
        width = im.shape[1]
        height = im.shape[0]
        dim = (width*3, height*3)
        image_vis = cv2.resize(im,dim)

        image_vis, flag, s = cls.sizeEstimation(image_vis, flag)
        cv2.imshow('out_fname',image_vis)
        cv2.waitKey(1)
        #-----------------------------------------------------------------------------------------------------------
        #logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "entry_idx": 0,
        }
        return context


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    args.input = 'projects/DensePose/photo1.jpg'
    #args = parser.parse_args(['show', 'configs\densepose_rcnn_R_50_FPN_DL_s1x.yaml', 'densepose_rcnn_R_50_FPN_s1x.pkl',  'photo.jpg','dp_iuv_texture', '--texture_atlas','texture_from_SURREAL.png', '-v'])
    #args = parser.parse_args([--verbosity='1', cfg='projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml', model='projects/DensePose/densepose_rcnn_R_50_FPN_s1x.pkl', input='projects/DensePose/photo.jpg', opts=[], visualizations='dp_iuv_texture', min_score=0.8, nms_thresh=None, texture_atlas='projects/DensePose/texture_from_SURREAL.png', texture_atlases_map=None, output='outputres.png'])
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(verbosity_to_level(verbosity))

    args.func(args)


if __name__ == "__main__":
    main()
