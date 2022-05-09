import sys
import requests
import tarfile
from os import path
from PIL import Image
from PIL import ImageFont, ImageDraw
from glob import glob
from matplotlib import pyplot as plt

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import argparse
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

# Import some common libraries

import numpy as np
import os, json, cv2, random

# Import some common detectron2 utilities

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from detectron2.structures import BoxMode
import yaml
from detectron2.data.datasets import register_coco_instances

def infer_layout(img_path, layout_model, confidence, output_dir):
  custom_config = "custom_labels_weights.yml"
  with open(custom_config, 'r') as stream:
      custom_yml_loaded = yaml.safe_load(stream)

  config_list = list(custom_yml_loaded['WEIGHT_CATALOG'].keys()) + list(custom_yml_loaded['MODEL_CATALOG'].keys())

  config_filePath = "configs/layout_parser_configs"
  config_name = layout_model

  # Capture model weights

  if config_name.split('_')[0] == 'Sanskrit':
      core_config = config_name.replace('Sanskrit_', '')
      config_file = config_filePath + '/' + custom_yml_loaded['MODEL_CATALOG'][core_config]
      model_weights = custom_yml_loaded['WEIGHT_CATALOG'][config_name]
      label_mapping = custom_yml_loaded['LABEL_CATALOG']["Sanskrit_Finetuned"]

  else:
      config_file = config_filePath + '/' + custom_yml_loaded['MODEL_CATALOG'][config_name]
      yaml_file = open(config_file)
      parsed_yaml_file = yaml.load(yaml_file, Loader = yaml.FullLoader)
      model_weights = parsed_yaml_file['MODEL']['WEIGHTS']
      dataset = config_name.split('_')[0]
      label_mapping = custom_yml_loaded['LABEL_CATALOG'][dataset]

  label_list = list(label_mapping.values())

  # Setting the confidence threshold

  confidence_threshold = confidence

  # Set custom configurations

  cfg = get_cfg()
  cfg.merge_from_file(config_file)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold # set threshold for this model
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list(label_mapping.keys()))

  print("ROI Heads is taken as",cfg.MODEL.ROI_HEADS.NUM_CLASSES)

  cfg.MODEL.WEIGHTS =  model_weights

  # Get predictions

  predictor = DefaultPredictor(cfg)
  im = cv2.imread(img_path)
  im_name = img_path.split("/")[-1]
  cv2.imwrite(f"{output_dir}/{im_name}", im)
  outputs = predictor(im)
  #print(outputs["instances"].pred_classes)
  #print(outputs["instances"].pred_boxes)
  
  # Save predictions

  dataset_name = 'data'
  DatasetCatalog.clear()
  MetadataCatalog.get(f"{dataset_name}_infer").set(thing_classes=label_list)
  layout_metadata = MetadataCatalog.get(f"{dataset_name}_infer")
  #print("Metadata is ",layout_metadata)

  v = Visualizer(im[:, :, ::-1],
                      metadata=layout_metadata,
                      scale=0.5
        )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  ans = out.get_image()[:, :, ::-1]
  im = Image.fromarray(ans)
  img_name = 'image_with_predictions.jpg'
  im.save(f"{output_dir}/{img_name}")

  # extracting, bboxes, scores and labels

  img = Image.open(img_path)
  instances = outputs["instances"].to("cpu")
  boxes = instances.pred_boxes.tensor.tolist()
  scores = instances.scores.tolist()
  labels = instances.pred_classes.tolist()
  layout_info = {}

  count = {}
  for i in range(len(label_list)):
    count[label_list[i]] = 0

  for score, box, label in zip(scores, boxes, labels):
    x_1, y_1, x_2, y_2 = box
    label_name = label_mapping[label]
    count[label_name] += 1
    l_new = label_name+str(count[label_name])
    info_data = {"box":box, "confidence": score}
    layout_info[l_new] = info_data
    #print(str(l_new) + ":",box)

  # storing the labels and corresponding bbox coordinates in a json
  layout_info_sort = {k: v for k, v in sorted(layout_info.items(), key=lambda item: item[1]["box"][1], reverse=True)}
  
  with open(f"{output_dir}/layout_data.json", 'w', encoding='utf-8') as f:
    json.dump(layout_info_sort, f, ensure_ascii=False, indent=4)

  return img, layout_info
  
if __name__ == "__main__":
    infer_layout()
    
