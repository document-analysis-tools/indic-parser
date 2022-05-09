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




