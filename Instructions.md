
GETTING STARTED - INITIAL SET UP - RUNNING PREDICTION ON A MODEL
##########################################################################################################
1. create a virtual env with python 3.9.12   conda create ....
2. install Cuda toolkit 11.3.1 using  conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit
3. Run from terminal:  pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html #Make sure all previous version are uninstalled#
4. Install Detectron2: python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

	Install: opencv and opencv-jupyter-ui if they are not already.

	pip install opencv-python 
	pip install opencv-jupyter-ui
	
	#It is helpful to switch to a notebook after this step. You may want to use as areference 'instruction_steps.ipynb'

5. Clone AdelaiDET - !git clone https://github.com/aim-uofa/AdelaiDet.git 
6. cd to AdelaiDet and run !python setup.py build develop

7. pip install opencv-python
8. To test whether AdelaiDet and Detectron2 are in order, object detection and instance segmentation predictions can be made.  You can startcheck with the following:
	1. Inference with Pre-trained Models of FCOS 
		Pick a model and its config file, for example, fcos_R_50_1x.yaml.
		Download the model using:  !wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O fcos_R_50_1x.pth #the model to make OD predictions on
		Run the demo with:
		
		!python demo/demo.py \
		--config-file configs/FCOS-Detection/R_50_1x.yaml \
		--input input1.jpg input2.jpg \
		--opts MODEL.WEIGHTS fcos_R_50_1x.pth
#NOTE remove '!' at the begininning ofthe commands if you are not using jupyter notebook or colab
 	
	2. Inference with Pre-trained Models of BlendMask on COCO dataset on any image of your choice
		Download the model using: wget -O blendmask_r101_dcni3_5x.pth https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download #the model to make IS predictions on
		Run the demo with:
		
		python demo/demo.py \
		--config-file configs/BlendMask/R_101_dcni3_5x.yaml \
		--input path_to_image \
		--confidence-threshold 0.35 \
		--opts MODEL.WEIGHTS blendmask_r101_dcni3_5x.pth
If everything works at this stage, then an existing model can be used to perform predictions.
 
9. To run the demo for our own model i.e BlendMask model for the Vaccinium dataset,we need the following different from above:
	1. config-file  file (configs/BlendMask/R_50_1x.yaml) 
	2. the model_final.pth of our model e.g 'Vaccc_train/model_final.pth'
	3. confidence threshold typically 0.3
	4. register the dataset in demo.py to show names of classes after prediction. In the syntax above, demo.py can be reaplced with demo_vacc.py(file is provided) and then you dont have to do any other thing.
	5. edit Base-BlendMask.yaml to have the registered dataset in demo.py/demo.vacc.py under DATASETS:  TRAIN: ("file_name",)TEST: ("file_name",). #If this is not done well, only numbers appear instead of class names
		 While only TEST is required for prediction. it is advised to have the registered name for TRAIN set too as this Base-BlendMask.yaml also references the training script (train-net.py) for the registered datasets. 
		 (Base-BlendMask with corresponding registered dataset name with the ones in the demo_vacc.py is provided)

	For instance, the commands could be as follows:
	
	!python demo/demo_vacc.py \
	--config-file configs/BlendMask/R_50_1x.yaml \
	--input 'vacc_plant.png'\
	--confidence-threshold 0.3 \
	--opts MODEL.WEIGHTS path_to_model_dir/model_final.pth     #check a new window on task bar to view predictions


#NOTE: I have to use demo_vacc.py because of object classes metadata registration required by the framework. The models 
##########################################################################################################


TRAINING THE NETWORK
##########################################################################################################
DEPENDENCIES: 
			Warning/ 											SOLUTION
tensorboard 2.9.1 requires requests<3,>=2.21.0, which is not installed.	/			pip install requests=2.28.1
tensorboard 2.9.1 requires werkzeug>=1.0.1, which is not installed.	/			pip install werkzeug==2.1.2
scikit-learn 1.1.1 requires scipy>=1.3.2, which is not installed.	/			pip install scipy==1.9.0rc1
pycocotools 2.0.4 requires matplotlib>=2.1.0, which is not installed.	/			pip install matplotlib==3.5.2
fvcore 0.1.5.post20220512 requires tqdm, which is not installed.	/			pip install tqdm==4.64.0


Downgrade the following to:
rapidfuzz to 2.1.1 ,
numpy to 1.23.0 ,
setuptools to 59.5.0 ,
shapely==1.8.2 



##########################################################################################################

TRAINING ON A CUSTOM DATASET
##########################################################################################################
To train a custom dataset.

1. To train a dataset, steps 1 to 8 above are very important before moving ahead.

2. Set up the directory and file hierarchy for the datasets as shown below. #NOTE: the cloned repo only contains AdelaiDet/datasets. So a dir 'coco' is created.'
		AdelaiDet/datasets/coco 
				coco should contain the folders annotations, train2017, val2017, test2017 
				# annotation contains the annotation json files of train, val and test datasplit
				# train2017, val2017, and test2017 contain raw images of the train, validation and test datsets respectively.

3. run the command: 'python datasets/prepare_thing_sem_from_instance.py', to extract semantic labels from instance annotations. #NOTE the script 'prepare_thing_sem_from_instance.py' comes with the repo
   After running he command, a directory  named thing2017 is craeted in the coco directory. This directory contains .npz files of the images in the train dataset.

4. Edit configs/BlendMask/Base-BlendMask.yaml to reflect the desired Batch size, number of iterations and learning rate.

5. If you are using just one GPU, edit adet/config/defaults.py and change the string SynBN to BN i _C.MODEL.BASIS_MODULE.NORM = "BN" (line 141). File is also 	provided(defaults.py).

6. run the following commands to train your network:
	!OMP_NUM_THREADS=1 python tools/train_net.py \      
    	--config-file configs/BlendMask/R_50_1x.yaml \
    	--num-gpus 1 \
    	OUTPUT_DIR training_dir/blendmask_R_50_1x


#NOTE: Pay attention to the names and paths of the training script, config file and directory wher the model will be saved, and the numberof GPU being used. 

train_net.py is the main training script. This script must be customized for the dataset. The datset must be registered and the registered name must be specicified in AdelaiDet/configs/BlendMask/Base-BlendMask.yaml.
For simplicity the customized dataset has been provided 'train_net_vacc.py'. It can be renamed. 

Using the original train_net.py from the original repo, returns: "AssertionError: Attribute 'thing_classes' in the metadata of 'coco_2017_train' cannot be set to a different value!..."
##########################################################################################################



POSSIBLE DEPENDENCY PROBLEMS AND SOLUTIONS
##########################################################################################################
PROBLEM:site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module> LooseVersion = distutils.version.LooseVersion AttributeError: module 'distutils' has no attribute 'version'
SOLUTION: Downgrade setuptools to 59.5.0

PROBLEM:AttributeError: module 'numpy' has no attribute 'bool'. `np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. 
If you specifically wanted the numpy scalar type, use `np.bool_` here.
SOLUTION: Downgrade to numpy 1.23.1

PROBLEM: "Multi-polygon object not iterable"
SOLUTION: DOwngrade 'shapely' e.g.  pip install shapely==1.8.2
##########################################################################################################



SUPPLEMENTARY UTILITY FUNCTIONS
##########################################################################################################
There is an utility python file that has functions to do tasks such as plot ground truth of images with annotations.The use of these supplemetary functions are illustrated in 'utilis_illustrate.ipynb' 

The functions description are as follows

1. plot_samples(dataset, number_of_samples) #This plots random ground truth and class labels of images in a registered dataset, 

2. plot_spec_anno_ (registered_dataset, img_name) #For a known specific image, in a dataset that has been registered, the ground truth annotations can be plotted usinf the function and the arguments

3. plot_groundt_anno (registered_dataset_name, dir_name) #To plot groud truth annotations of all images ina dataset and saved in the directory with name dir_nam
#dataset  must have been registered  with the name 'dataset_name'

3. The use of these supplemetary functions are illustrated in 'utilis_illustrate.ipynb'

#NOTE to use these functions, it is helpful to:
1. have paths of json annotations and raw images assigned to string variables
2. register the dataset you maybe interested in
3. To use the prediction functions, in addition to 1. and 2, assign class definitions to a string variable

##########################################################################################################




