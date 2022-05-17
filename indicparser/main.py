
import layoutparser as lp
import pandas as pd
import numpy as np
import cv2
import os
from os import path
try:
 from PIL import Image
except ImportError:
 import Image
import pytesseract
from pdf2image import convert_from_path
import sys
from pdfreader import SimplePDFViewer
import subprocess
import json
from pathlib import Path
from uuid import uuid4
from math import floor
# Execute layout inference

import requests
import tarfile
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
import random

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






# from layout_inference import infer_layout

#does the user want to use layout inference ? 
# infer_flag = input("Do you wish to use Layout Inference? (yes or no)")

#initialise language model
class model:
  def __init__(self,config_path,lang_model=None):
    '''
    param config_path (str) :  the configuration file path
    param lang_model (int) : key for the _tesslanglist

    _tessdata_dir_config (prev. tessdata_dir_config): config path for pytesseract.get_languages(), to get the languages
    __lcount : ?
    _languages (prev. languages) : all the different languages available for the given config taken from  _tessdata_dir_config
    _lang_model (prev. int(linput)-1) : lang_model
    _output_dir (prev. output_dir) : the directory where the HOCR output would be saved
    _lang : language selected or the value corresponding to _lang_model in _tess_langlist dict
    '''

    self._tessdata_dir_config = r'--tessdata-dir ' + config_path            #= r'--tessdata-dir "/content/layout-with-ocr/configs/tessdata"' #must change while running locally
    # self.__lcount=0 
    self._languages=pytesseract.get_languages(config=self._tessdata_dir_config)
    self._tesslanglist = {}
    self._lang_model = lang_model
    self._output_dir = ''
    self._lang = ''
  os.environ["TESSDATA_PREFIX"] = str(self._tessdata_dir_config) #'/content/layout-with-ocr/configs/tessdata'
  
  

  for idx,l in enumerate(self._languages):
    # if not (l== 'osd'):
      self._tesslanglist[idx] = l                        #[self.__lcount]=l
      # lcount+=1
      print(str(idx)+'. '+l)                             # idx was self.__lcount

# linput=input("Choose the language model for OCR from the above list: ")
  
  
  def _language_models(self):
    '''
    prints all the available language models
    '''
    print(self._tesslanglist)

  # to print the model selected
  def _language(self):
    '''
    prints the language selected
    '''
    self._lang = self._tesslanglist[self._lang_model]
    print('Selected language model: '+ self._lang)



  '''
  Add this in the trace back

  if not (int(linput)-1) in tesslanglist:
    print("Not a correct option! Exiting program")
    sys.exit(1)
  '''
# input_lang is now self._lang

  #initialise output directory 
  def make_output_directory(self,output_dir):
    try:
      # output_dir = input("Directory for OCR output: \n")
      if(output_dir.find(" ")!=-1):
        raise NameError("File name contains spaces")
      else:
        self._output_dir = output_dir
    except Exception as err:
      print("Error: {0}".format(err))
      sys.exit(1)

    if not os.path.exists(self._output_dir):
      os.mkdir(self._output_dir) 



  __ocr_agent = lp.TesseractAgent(languages=self._lang)

  def _get_LEVELS(self,per_level):
    return self.__LEVELS[per_level]

  __LEVELS = {
      'page_num': 1,
      'block_num': 2,
      'par_num': 3,
      'line_num': 4,
      'word_num': 5
      }


class preprocesing(model):
  def __init__(self):
    self.__filename = ''
    self.__image_width = 0
    self.__image_height = 0



  def create_image_url(self,filepath):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    self.__filename = os.path.basename(filepath)
    return f'http://localhost:8081/{self.__filename}'

  def convert_to_ls(self,image, tesseract_output, per_level='block_num'):
    """
    :param image: PIL image object
    :param tesseract_output: the output from tesseract
    :param per_level: control the granularity of bboxes from tesseract
    :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template
    """
    self.__image_width, self.__image_height = image.size
    per_level_idx = super()._get_LEVELS(per_level)         # getting a LEVEL from super class model
    results = []
    all_scores = []
    for i, level_idx in enumerate(tesseract_output['level']):
      if level_idx == per_level_idx:
        bbox = {
          'x': 100 * tesseract_output['left'][i] / self.__image_width,
          'y': 100 * tesseract_output['top'][i] / self.__image_height,
          'width': 100 * tesseract_output['width'][i] / self.__image_width,
          'height': 100 * tesseract_output['height'][i] / self.__image_height,
          'rotation': 0
        }

        words, confidences = [], []
        for j, curr_id in enumerate(tesseract_output[per_level]):
          if curr_id != tesseract_output[per_level][i]:
            continue
          word = tesseract_output['text'][j]
          confidence = tesseract_output['conf'][j]
          words.append(word)
          if confidence != '-1':
            confidences.append(float(confidence / 100.))

        text = ' '.join((str(v) for v in words)).strip()
        if not text:
          continue
        region_id = str(uuid4())[:10]
        score = sum(confidences) / len(confidences) if confidences else 0
        bbox_result = {
          'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
          'value': bbox}
        transcription_result = {
          'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
          'value': dict(text=[text], **bbox), 'score': score}
        results.extend([bbox_result, transcription_result])
        all_scores.append(score)

    return {
      'data': {
        'ocr': create_image_url(image.filename)
      },
      'predictions': [{
        'result': results,
        'score': sum(all_scores) / len(all_scores) if all_scores else 0
      }]
    }

  def infer_layout(self,output_dir):
    custom_config = "custom_labels_weights.yml"
    with open(custom_config, 'r') as stream:
        custom_yml_loaded = yaml.safe_load(stream)

    config_list = list(custom_yml_loaded['WEIGHT_CATALOG'].keys()) + list(custom_yml_loaded['MODEL_CATALOG'].keys())
    print("config_list is ",config_list)

    config_filePath = "configs/layout_parser_configs"
    index = 1
    config_filesDict = {}
    for cfile in config_list:
        config_filesDict[index] = cfile
        print(index,":",cfile)
        index+=1

    print(" ")
    chosenFile = input("choose the model for the inference : ")
    print("Selected Model = ",config_filesDict[int(chosenFile)])

    print(" ")
    
    config_name = config_filesDict[int(chosenFile)]
    print(config_name.split('_')[0] == 'Sanskrit')

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

    print("model weights fetched :",model_weights)

    # Choosing the image for the inference

    input_image_path = 'test_img/'
    input_choice = input("Choose a random image (yes/no) : ")
    isRandom = True
    if input_choice == 'no':
        isRandom = False
    if isRandom == False:
        image_name = input("Enter the image name : ")
        input_image_path = input_image_path + image_name
    else:
        random_image_name = random.choice(os.listdir(input_image_path))
        input_image_path = input_image_path + random_image_name

    print("Selected image = ",input_image_path)
    print(" ")

    # Setting the confidence threshold

    confidence_threshold = float(input("Set the confidence threshold, choose from 0 to 1 (eg: 0.7) : "))
    print(" ")

    # Set custom configurations

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list(label_mapping.keys()))

    print("ROI Heads is taken as",cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    cfg.MODEL.WEIGHTS =  model_weights

    # Get predictions

    predictor = DefaultPredictor(cfg)
    im = cv2.imread(input_image_path)
    outputs = predictor(im)
    #print(outputs["instances"].pred_classes)
    #print(outputs["instances"].pred_boxes)
    
    # Save predictions

    dataset_name = 'data'
    DatasetCatalog.clear()
    MetadataCatalog.get(f"{dataset_name}_infer").set(thing_classes=label_list)
    layout_metadata = MetadataCatalog.get(f"{dataset_name}_infer")
    print("Metadata is ",layout_metadata)

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

    img = Image.open(input_image_path)
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


class hocr(preprocesing):
  def __init__(self):
    pass


  def create_hocr(self,img_dir=None, languages=None, linput=None, output_path=None,img_from_inference=None,layout_info=None):
    '''
    param img_dir : path of directory of pdfs or jpegs

    '''
    if layout_info == None:
      if os.path.isdir(img_dir):
        for img_file in os.listdir(img_dir):
          if img_file.endswith('.pdf'):
            print("OCR-ing pdfs...\n")
            newdir = output_dir + "/" + img_file.replace(".pdf", "")
            os.mkdir(newdir)
            os.mkdir(newdir + "/page_images")
            os.mkdir(newdir + "/output")
            img_path= img_dir + "/" + img_file
            print("Converting to images...\n")
            convert_from_path(img_path,
                output_folder= newdir + "/page_images",
                paths_only=True,
                fmt='jpg',
                output_file="O",
                use_pdftocairo=True,
              )
            tasks = []
            for img_ in os.listdir(newdir + "/page_images"):
              print(img_)
              #image = cv2.imread(newdir + "/page_images/" + img_)
              image = Image.open(newdir + "/page_images/" + img_)

              img_path = newdir + "/page_images/" + img_
              output_path = output_dir + '/' + img_[:-4]
              pytesseract.pytesseract.run_tesseract(img_path, output_path, extension="jpg", lang=languages[linput], config="--psm 4 -c tessedit_create_hocr=1")

              res = ocr_agent.detect(image, return_response = True)
              tesseract_output = res["data"].to_dict('list')
              with open(newdir + "/output/" + img_[:-4] + '.txt', 'w') as f:
                f.write(res["text"])
              task = super().convert_to_ls(image, tesseract_output, per_level='block_num')
              tasks.append(task)
              with open("./" + newdir + "/output/" + img_[:-4] + '_ocr_tasks.json', mode='w') as f:
                json.dump(task, f, indent=2)

          elif img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg'):
            print("OCR-ing images...\n")
            #image = cv2.imread(img_dir + "/" + img_file)
            image = Image.open(img_dir + "/" + img_file)

            img_path = img_dir + "/" + img_file
            if img_file.endswith('.jpeg'):
              x = img_file[:-5]
            else:
              x = img_file[:-4]
            
            output_path = output_dir + '/' + x
            pytesseract.pytesseract.run_tesseract(img_path, output_path, extension="jpg", lang=languages[linput], config="--psm 4 -c tessedit_create_hocr=1")

            res = ocr_agent.detect(image, return_response = True)
            tesseract_output = res["data"].to_dict('list')
            tasks = []
            if img_file.endswith('.jpeg'):
              x = img_file[:-5]
            else:
              x = img_file[:-4]
            with open(output_dir + '/' + x + '.txt', 'w') as f:
              f.write(res["text"])
            task = super().convert_to_ls(image, tesseract_output, per_level='block_num')
            tasks.append(task)
            with open(output_dir + '/' + x + '_ocr_tasks.json', mode='w') as f:
              json.dump(tasks, f, indent=2)

    else:

      hocr_data = {}
      layout_info_sort = {k: v for k, v in sorted(layout_info.items(), key=lambda item: item[1]["box"][1], reverse=True)}
      with open(f'{output_dir}/output-ocr.txt', 'w') as f:
        for label, info_dict in layout_info_sort.items():
          img_cropped = img.crop(info_dict["box"])
          res = ocr_agent.detect(img_cropped)
          f.write(res)
          hocr_data[res] = layout_info_sort[label]
        f.close()

      hocr_sorted_data = {k: v for k, v in sorted(hocr_data.items(), key=lambda item: item[1]["box"][1], reverse=True)}
      with open(f"{output_dir}/hocr_data.json", 'w', encoding='utf-8') as f:
        json.dump(hocr_sorted_data, f, ensure_ascii=False, indent=4)
      
      print("OCR is complete. Please find the output in the provided output directory.")

      f = open(f'{output_dir}/layout.hocr', 'w+')
      header = '''
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
      <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
      <head>
        <title></title>
        <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
        <meta name='ocr-system' content='tesseract v5.0.1.20220118' />
        <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf'/>
      </head>
      <body>
        <div class='ocr_page' id='page_1'>
      '''
      f.write(header)
      f.close()
      for i, item in enumerate(list(hocr_sorted_data.items())):
        hocr_block(item[0], hocr_sorted_data, i)

      footer = ['  </div>\n',' </body>\n','</html>\n']
      f = open(f'{output_dir}/layout.hocr', 'a')
      f.writelines(footer)
      f.close()

    print("OCR is complete. Please find the output in the provided output directory.")
    

  def hocr_block(self,k, hocr_sorted_data, i):
    carea = f'''   <div class='ocr_carea' id='block_1_{i+1}'>\n'''
    par = f'''    <p class='ocr_par' id='par_1_{i+1}' lang='san'>\n'''
    bbox = " ".join([str(floor(value)) for value in hocr_sorted_data[k]["box"]])
    conf = str(floor(hocr_sorted_data[k]["confidence"] * 100))
    line = f'''     <span class='ocr_line' id='line_1_{i+1}' title="bbox {bbox}; x_conf {conf}">\n'''
    words = k.strip().split(" ")
    word_list = []
    for n,w in enumerate(words):
      word_list.append(f'''      <span class='ocrx_word' id='word_1_{n+1}'>{w}</span>\n''')
    
    f = open(f'{output_dir}/layout.hocr', 'a')
    l = [carea, par, line]
    f.writelines(l)
    f.writelines(word_list)
    f.writelines(['     </span>\n','    </p>\n','   </div>\n'])
    f.close()
