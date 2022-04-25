# Digitization for Documents in Indian Languages 

A repository that combines OCR with custom Layout Detection for Sanskrit and English documents. 
Simply upload your images or pdfs for OCR in the 'test images' folder of this repository, choose a model for OCR and Layout Detection, and recieve the OCR output and infered image in your output folder! 

To run lp_ocr.py, wrapper for Layout Parser OCR **for the first time**: 
- Create a venv and activate:  
1. virtualenv lp_ocr
2. source lp_ocr/bin/activate
- Install all packages in the environment: 
1. pip3 install -r requirements.txt
2. apt install tesseract-ocr
3. apt install libtesseract-dev
4. apt-get install poppler-utils

- For layout detection and OCR:   
    - Run lp_ocr.py and select 'yes' when asked if layout detection should be applied
    - Choose a custom layout model. eg. Choose a Sanskrit model if your image/pdf is in an Indic language. 
    - Choose an OCR model. 
    - Define your output folder name and find the OCR'd text + a Label Studio formatted output retrieved from infered bounding boxes from the Layout Detection model. 
- For document layout analysis of an image: 
    - Run layout_inference.py. This will return an infered image with masks and a json file with layout data - bounding boxes. 
- For OCR of a directory of images: 
    - Run lp_ocr.py and select 'no' when asked if layout detection should be applied, and supply your input image directory. 


