
GETTING STARTED - INITIAL SET UP - RUNNING PREDICTION ON A MODEL
###################################################################################################################################################################################################################################
1. create a virtual env with python 3.9.12
2. install Cuda toolkit 11.3.1 using  conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit
3. Run from terminal:  pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html #Make sure all previous version are uninstalled#
4. Install Detectron2: python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
5. Clone AdelaiDET - !git clone https://github.com/aim-uofa/AdelaiDet.git
6. cd to AdelaiDet and run !python setup.py build develop

7. pip install opencv-python
8. To test whether AdelaiDet and Detectron2 are in order, object detection and instance segmentation predictions can be made.  You can startcheck with the following:
	1. Inference with Pre-trained Models of FCOS 
		Pick a model and its config file, for example, fcos_R_50_1x.yaml.
		Download the model using:  wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O fcos_R_50_1x.pth #the model to make OD predictions on
		Run the demo with
			python demo/demo.py \
    			--config-file configs/FCOS-Detection/R_50_1x.yaml \
    			--input input1.jpg input2.jpg \
    			--opts MODEL.WEIGHTS fcos_R_50_1x.pth

 	2. Inference with Pre-trained Models of BlendMask on COCO dataset on any image of your choice
		Download the model using: wget -O blendmask_r101_dcni3_5x.pth https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download #the model to make IS predictions on
		Run the demo with:
			python demo/demo.py \
    			--config-file configs/BlendMask/R_101_dcni3_5x.yaml \
    			--input datasets/coco/val2017/000000005992.jpg \
    			--confidence-threshold 0.35 \
    			--opts MODEL.WEIGHTS blendmask_r101_dcni3_5x.pth
If everything works at this stage, then an existing model can be used to perform predictions.
 
10. To run the demo for our own model i.e BlendMask model for the Vaccinium dataset,we need the following different from above:
		1. config yaml  file
		2. the model_final.pth of our model e.g 'Vaccc_train/model_final.pth'
		4. register the dataset in demo.py to show names of classes after prediction. In the syntax above, demo.py can be reaplced with demo_vacc.py(file is provided) and then you dont have to do any other thing.
		3. edit Base-BlendMask.yaml to have the registered dataset in demo.py/demo.vacc.py under DATASETS:  TRAIN: ("file_name",)TEST: ("file_name",). #If this is not done well, only numbers appear instead of class names
		 While only TEST is required for prediction. it is advised to have the registered name for TRAIN set too as this Base-BlendMask.yaml also references the training script (train-net.py) for the registered datasets. 
		 (Base-BlendMask with corresponding registered dataset name with the ones in the demo_vacc.py is provided)

	For instance, the commands could be as follows:
		!python demo/demo_vacc.py \
		--config-file configs/BlendMask/R_50_1x.yaml \
    		--input 'vacc_plant.png'\
    		--confidence-threshold 0.3 \
	    	--opts MODEL.WEIGHTS vaccinium_blendmask_R_50_1x/model_final.pth     #check a new window on task bar to view predictions


#NOTE: I have to use demo_vacc.py because of object classes metadata registration required by the framework
###################################################################################################################################################################################################################################



TRAINING ON A CUSTOM DATASET
###################################################################################################################################################################################################################################
To train a custom dataset.

1. To train a dataset, steps 1 to 9 above are very important before moving ahead.

2. Set up the directory and file hierarchy for the datasets as shown below. #NOTE: the cloned repo only contains AdelaiDet/datasets. So a dir 'coco' is created.'
		AdelaiDet/datasets/coco 
				coco should contain the folders annotations, train2017, val2017, test2017 
				# annotation contains the annotation json files of train, val and test datasplit
				# train2017, val2017, and test2017 contain raw images of the train, validation and test datsets respectively.

3. run the command: 'python prepare_thing_sem_from_instance.py', to extract semantic labels from instance annotations. #NOTE the script 'prepare_thing_sem_from_instance.py' comes with the repo
   After running he command, a directory  named thing2017 is craeted in the coco directory. This directory contains .npz files of the images in the train dataset.

4.Edit configs/BlendMask/Base-BlendMask.yaml to reflect the desired Batch size, number of iterations and learning rate
5.If you are using just one GPU, edit adet/config/defaults.py and change the string SynBN to BN i _C.MODEL.BASIS_MODULE.NORM = "BN" (line 141)

6. run the following commands to train your network:
	!OMP_NUM_THREADS=1 python tools/train_net.py \      
    	--config-file configs/BlendMask/R_50_1x.yaml \
    	--num-gpus 1 \
    	OUTPUT_DIR training_dir/blendmask_R_50_1x


#NOTE: Pay attention to the names and paths of the training script, config file and directory wher the model will be saved, and the numberof GPU being used 

train_net.py is the main training script. This script must be customized for the dataset. The datset must be registered and the registered name must be specicified in AdelaiDet/configs/BlendMask/Base-BlendMask.yaml.
For simplicity the customized dataset has been provided 'train_net_vacc.py'. It can be renamed. 

Using the original train_net.py from the original repo, returns: AssertionError: Attribute 'thing_classes' in the metadata of 'coco_2017_train' cannot be set to a different value!...
#########################################################################################################################################################################################################################################



POSSIBLE DEPENDENCY PROBLEMS AND SOLUTIONS
#######################################################################################################################################################################################################################################
PROBLEM:site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module> LooseVersion = distutils.version.LooseVersion AttributeError: module 'distutils' has no attribute 'version'
SOLUTION: Downgrade setuptools to 59.5.0

PROBLEM:AttributeError: module 'numpy' has no attribute 'bool'. `np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. 
If you specifically wanted the numpy scalar type, use `np.bool_` here.
SOLUTION: Downgrade to numpy 1.23.1

PROBLEM: "Multi-polygon object not iterable"
SOLUTION: DOwngrade 'shapely' e.g.  pip install shapely==1.8.2
###################################################################################################################################################################################################################################



SUPPLEMENTARY UTILITY FUNCTIONS
#########################################################################################################################################################################################################################################
There is an utility python file that has functions to do tasks such as plot ground truth of  images with annotations. The functions are as follows

1. plot_samples(dataset, number_of_samples) #This plots random images in a dataset, s

2. plot_spec_anno_ (registered_dataset, img_name) #For a known specific image, in a dataset that has been registered, the ground truth annotations can be plotted usinf the function and the arguments

3. plot_groundt_anno (registered_dataset_name, dir_name) #To plot groud truth annotations of all images ina dataset and saved in the directory with name dir_nam
#dataset  must have been registered  with the name 'dataset_name'

3. The use of these supplemetary functions are illustrated in 'utilis_illustrate.ipynb'

#NOTE to use these functions, it is helpful to:
1. have paths of json annotations and raw images assigned to string variables
2. register the dataset you maybe interested in
3. To use the prediction functions, in addition to 1. and 2, assign class definitions to a string variable

#########################################################################################################################################################################################################################################



INSTALLED PACKAGES IN MY VIRTUALENV
#########################################################################################################################################################################################################################################
Python Version: 3.9.12

pip list:

Package                      Version            Editable project location
---------------------------- ------------------ ---------------------------------------------------------------
absl-py                      1.1.0
AdelaiDet                    0.2.0              /home/xxx/xxx/xxx/AdelaiDet
antlr4-python3-runtime       4.9.3
anyio                        3.5.0
appdirs                      1.4.4
argon2-cffi                  21.3.0
argon2-cffi-bindings         21.2.0
asttokens                    2.0.5
astunparse                   1.6.3
attrs                        22.1.0
backcall                     0.2.0
beautifulsoup4               4.12.2
black                        21.4b2
bleach                       4.1.0
cachetools                   5.2.0
certifi                      2022.12.7
cffi                         1.15.1
charset-normalizer           3.1.0
click                        8.1.3
cloudpickle                  2.2.1
comm                         0.1.2
contourpy                    1.0.7
cycler                       0.11.0
debugpy                      1.5.1
decorator                    5.1.1
defusedxml                   0.7.1
detectron2                   0.6+cu113          /home/xxx/.conda/envs/py3912/lib/python3.9/site-packages
editdistance                 0.6.2
entrypoints                  0.4
executing                    0.8.3
fairscale                    0.4.6
fastjsonschema               2.16.2
flatbuffers                  1.12
fonttools                    4.39.3
future                       0.18.3
fvcore                       0.1.5.post20220512
gast                         0.4.0
google-auth                  2.9.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.47.0
h5py                         3.8.0
hydra-core                   1.2.0
idna                         3.4
imageio                      2.27.0
importlib-metadata           6.0.0
importlib-resources          5.12.0
iopath                       0.1.9
ipycanvas                    0.13.1
ipykernel                    6.19.2
ipython                      8.12.0
ipython-genutils             0.2.0
ipywidgets                   8.0.6
jarowinkler                  1.0.5
jedi                         0.18.1
Jinja2                       3.1.2
joblib                       1.1.0
jsonschema                   4.17.3
jupyter_client               8.1.0
jupyter_core                 5.3.0
jupyter-server               1.23.4
jupyter-ui-poll              0.2.2
jupyterlab-pygments          0.1.2
jupyterlab-widgets           3.0.7
keras                        2.9.0
Keras-Preprocessing          1.1.2
kiwisolver                   1.4.4
lazy_loader                  0.2
libclang                     16.0.0
lxml                         4.9.2
Markdown                     3.3.7
MarkupSafe                   2.1.1
matplotlib                   3.7.1
matplotlib-inline            0.1.6
mistune                      0.8.4
mock                         5.0.2
mypy-extensions              1.0.0
nbclassic                    0.5.5
nbclient                     0.5.13
nbconvert                    6.5.4
nbformat                     5.7.0
nest-asyncio                 1.5.6
networkx                     3.1
notebook                     6.5.4
notebook_shim                0.2.2
numpy                        1.23.1
oauthlib                     3.2.0
omegaconf                    2.2.2
opencv-jupyter-ui            1.4.2
opencv-python                4.7.0.72
opt-einsum                   3.3.0
packaging                    23.0
pandas                       2.0.0
pandocfilters                1.5.0
parso                        0.8.3
pathspec                     0.9.0
pexpect                      4.8.0
pickleshare                  0.7.5
Pillow                       9.5.0
pip                          23.0.1
platformdirs                 2.5.2
Polygon3                     3.0.9.1
portalocker                  2.4.0
prometheus-client            0.14.1
prompt-toolkit               3.0.36
protobuf                     3.19.4
psutil                       5.9.0
ptyprocess                   0.7.0
pure-eval                    0.2.2
pyasn1                       0.4.8
pyasn1-modules               0.2.8
pycocotools                  2.0.4
pycparser                    2.21
pydot                        1.4.2
Pygments                     2.11.2
pyparsing                    3.0.9
pyrsistent                   0.18.0
python-dateutil              2.8.2
pytz                         2023.3
PyWavelets                   1.4.1
PyYAML                       5.1
pyzmq                        23.2.0
rapidfuzz                    2.1.1
regex                        2023.3.23
requests                     2.28.2
requests-oauthlib            1.3.1
rsa                          4.8
scikit-image                 0.21.0rc0
scikit-learn                 1.1.1
scipy                        1.10.1
Send2Trash                   1.8.0
setuptools                   59.5.0
Shapely                      1.8.2
six                          1.16.0
sklearn                      0.0
sniffio                      1.2.0
soupsieve                    2.4
stack-data                   0.2.0
tabulate                     0.8.10
tensorboard                  2.9.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow-estimator         2.9.0
tensorflow-gpu               2.9.1
tensorflow-io-gcs-filesystem 0.32.0
termcolor                    1.1.0
terminado                    0.17.1
threadpoolctl                3.1.0
tifffile                     2023.4.12
timm                         0.5.4
tinycss2                     1.2.1
toml                         0.10.2
tomli                        2.0.1
torch                        1.10.1+cu113
torchaudio                   0.10.1+rocm4.1
torchvision                  0.11.2+cu113
tornado                      6.2
tqdm                         4.65.0
traitlets                    5.7.1
typing_extensions            4.5.0
tzdata                       2023.3
urllib3                      1.26.15
wcwidth                      0.2.5
webencodings                 0.5.1
websocket-client             0.58.0
Werkzeug                     2.2.3
wheel                        0.38.4
widgetsnbextension           4.0.7
wrapt                        1.15.0
yacs                         0.1.8
zipp                         3.11.0






