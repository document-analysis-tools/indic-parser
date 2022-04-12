import layoutparser as lp
import pandas as pd
import numpy as np
import cv2
import os
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

from layout_inference import infer_layout

#does the user want to use layout inference ? 
infer_flag = input("Do you wish to use Layout Inference? (yes or no)")

#initialise language model
tessdata_dir_config = r'--tessdata-dir "/content/layout-with-ocr/configs/tessdata"' #must change while running locally
os.environ["TESSDATA_PREFIX"] = '/content/layout-with-ocr/configs/tessdata'
languages=pytesseract.get_languages(config=tessdata_dir_config)
lcount=0 
tesslanglist={}
# print(languages)
for l in languages:
  # if not (l== 'osd'):
    tesslanglist[lcount]=l
    lcount+=1
    print(str(lcount)+'. '+l)

linput=input("Choose the language model for OCR from the above list: ")

if not (int(linput)-1) in tesslanglist:
  print("Not a correct option! Exiting program")
  sys.exit(1)

print("Selected language model: "+tesslanglist[int(linput)-1])

input_lang=tesslanglist[int(linput)-1]

#initialise output directory 
try:
  output_dir = input("Directory for OCR output: \n")
  if(output_dir.find(" ")!=-1):
    raise NameError("File name contains spaces")
except Exception as err:
  print("Error: {0}".format(err))
  sys.exit(1)

if not os.path.exists(output_dir):
  os.mkdir(output_dir) 

ocr_agent = lp.TesseractAgent(languages=input_lang)

LEVELS = {
    'page_num': 1,
    'block_num': 2,
    'par_num': 3,
    'line_num': 4,
    'word_num': 5
}

def create_image_url(filepath):
  """
  Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
  if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
  Otherwise you can build links like /data/upload/filename.png to refer to the files
  """
  filename = os.path.basename(filepath)
  return f'http://localhost:8081/{filename}'

def convert_to_ls(image, tesseract_output, per_level='block_num'):
  """
  :param image: PIL image object
  :param tesseract_output: the output from tesseract
  :param per_level: control the granularity of bboxes from tesseract
  :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template
  """
  image_width, image_height = image.size
  per_level_idx = LEVELS[per_level]
  results = []
  all_scores = []
  for i, level_idx in enumerate(tesseract_output['level']):
    if level_idx == per_level_idx:
      bbox = {
        'x': 100 * tesseract_output['left'][i] / image_width,
        'y': 100 * tesseract_output['top'][i] / image_height,
        'width': 100 * tesseract_output['width'][i] / image_width,
        'height': 100 * tesseract_output['height'][i] / image_height,
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

def create_hocr(image_path, languages, linput, output_path):
  pytesseract.pytesseract.run_tesseract(image_path, output_path, extension="jpg", lang=languages[linput], config="--psm 4 -c tessedit_create_hocr=1")

def hocr_block(k, hocr_sorted_data, i):
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
 

if infer_flag == "no":
  img_dir = input("Enter the name of our image folder for OCR: \n")
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
          create_hocr(img_path, languages, int(linput)-1, output_path)

          res = ocr_agent.detect(image, return_response = True)
          tesseract_output = res["data"].to_dict('list')
          with open(newdir + "/output/" + img_[:-4] + 'txt', 'w') as f:
            f.write(res["text"])
          task = convert_to_ls(image, tesseract_output, per_level='block_num')
          tasks.append(task)
        with open("./" + newdir + "/output/" + img_[:-4] + '_ocr_tasks.json', mode='w') as f:
          json.dump(tasks, f, indent=2)

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
        create_hocr(img_path, languages, int(linput)-1, output_path)

        res = ocr_agent.detect(image, return_response = True)
        tesseract_output = res["data"].to_dict('list')
        tasks = []
        if img_file.endswith('.jpeg'):
          x = img_file[:-5]
        else:
          x = img_file[:-4]
        with open(output_dir + '/' + x + '.txt', 'w') as f:
          f.write(res["text"])
        task = convert_to_ls(image, tesseract_output, per_level='block_num')
        tasks.append(task)
        with open(output_dir + '/' + x + '_ocr_tasks.json', mode='w') as f:
          json.dump(tasks, f, indent=2)
  print("OCR is complete. Please find the output in the provided output directory.")

elif infer_flag == "yes":
  img, layout_info = infer_layout(output_dir)
  #sorting layout_info by y_1 coordinate
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

else:
  print('Incorrect Input')
