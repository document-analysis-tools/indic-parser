import setuptools

setuptools.setup(
	name = 'indic-parser',
	version = '0.1',
	url = 'https://github.com/Saurabhbaghel/indicparser',
	packages = ['indicparser'],
	package_data = {
		'configs':['*.yaml'],
		},
	install_requires = [
		'numpy',
		'opencv-python',
		'torch==1.5', 
		'torchvision==0.6',
		'-f https://download.pytorch.org/whl/cu101/torch_stable.html',
		'pyyaml==5.4',
		'detectron2==0.1.3',
		'-f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html',
		'pytesseract',
		'pdf2image',
		'pdfreader',
		'layoutparser[ocr]'
		],
)
			
		
		
		
