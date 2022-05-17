# TableNet Inference

## How to Create the required Environment

1. Create a virtual environment with Python 3.8 and install pip
    ~~~~
    python get-pip.py
    ~~~~

2. Install the required libraries
    ~~~~
    pip install -r requirements1.txt
    ~~~~

3. Install torch(1.7.1) and torchvision(0.8.2)
    ~~~~
    pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/cu102/torch_stable.html
    ~~~~

## Run tablenet_inference.py 

1. By passing the path to the required model and image:
    ~~~~
    python tablenet_inference.py --model_weights='<weights path>' --image_path='<image path>'
    ~~~~
    
2. Run the file directly, to get output to sample image using standard weights:
    ~~~~
    python tablenet_inference.py
    ~~~~