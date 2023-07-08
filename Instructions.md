## GETTING STARTED - INITIAL SET UP - RUNNING PREDICTION ON A MODEL

1. Create a virtual environment with Python 3.9.12 using Conda:
conda create --name myenv python=3.9.12

2. Install CUDA Toolkit 11.3.1:
```conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit```


3. Install required PyTorch packages:
```pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html```

     Make sure to uninstall any previous versions of these packages.

5. Install Detectron2 and OpenCV:
```pip install detectron2 opencv-python opencv-jupyter-ui```
It is recommended to switch to a Jupyter Notebook after this step. You can refer to the `instruction_steps.ipynb` notebook for guidance.

6. Clone the AdelaiDet repository:
```git clone https://github.com/aim-uofa/AdelaiDet.git```

7. Navigate to the AdelaiDet directory and run the setup:
  cd AdelaiDet
```python setup.py build develop```

8. Install OpenCV (again): ```pip install opencv-python```

9. To test if AdelaiDet and Detectron2 are working correctly, you can make object detection and instance segmentation predictions using pre-trained models. You can start with the following examples:

	- Inference with Pre-trained Models of FCOS:
	  - Pick a model and its corresponding config file, for example, `fcos_R_50_1x.yaml`.
	  - Download the model:
	    ```
	    wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O fcos_R_50_1x.pth
	    ```
	  - Run the demo:
	    ```
	    python demo/demo.py \
	    --config-file configs/FCOS-Detection/R_50_1x.yaml \
	    --input input1.jpg input2.jpg \
	    --opts MODEL.WEIGHTS fcos_R_50_1x.pth
	    ```
	
	- Inference with Pre-trained Models of BlendMask on COCO dataset:
	  - Download the model:
	    ```
	    wget -O blendmask_r101_dcni3_5x.pth https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download
	    ```
	  - Run the demo:
	    ```
	    python demo/demo.py \
	    --config-file configs/BlendMask/R_101_dcni3_5x.yaml \
	    --input path_to_image \
	    --confidence-threshold 0.35 \
	    --opts MODEL.WEIGHTS blendmask_r101_dcni3_5x.pth
	    ```
	
	Note: Remove the `!` at the beginning of the commands if you are not using Jupyter Notebook or Colab.

10. If everything works correctly up to this point, you can use an existing model to perform predictions. To run the demo with the BlendMask model for the Vaccinium dataset, you need the following:
- A config file, e.g., `configs/BlendMask/R_50_1x.yaml`
- The `model_final.pth` file of your model, e.g., `Vaccc_train/model_final.pth`
- Confidence threshold (typically 0.3)
- Register the dataset in `demo.py` to show class names after prediction. You can use `demo_vacc.py` instead of `demo.py` for this step.
- Edit `Base-BlendMask.yaml` to include the registered dataset in `demo.py` or `demo_vacc.py` under `DATASETS: TRAIN: ("file_name",) TEST: ("file_name",)`. Make sure to reference the same dataset name in both files.
- Run the following command:
  ```
  python demo/demo_vacc.py \
  --config-file configs/BlendMask/R_50_1x.yaml \
  --input 'vacc_plant.png' \
  --confidence-threshold 0.3 \
  --opts MODEL.WEIGHTS path_to_model_dir/model_final.pth
  ```

  You will be able to view the predictions in a new window.

## TRAINING THE NETWORK

To train the network, you may encounter some dependency issues. Here are the possible problems and their solutions:

- Problem: `tensorboard 2.9.1 requires requests<3,>=2.21.0, which is not installed.`
Solution: Run `pip install requests==2.28.1`.

- Problem: `tensorboard 2.9.1 requires werkzeug>=1.0.1, which is not installed.`
Solution: Run `pip install werkzeug==2.1.2`.

- Problem: `scikit-learn 1.1.1 requires scipy>=1.3.2, which is not installed.`
Solution: Run `pip install scipy==1.9.0rc1`.

- Problem: `pycocotools 2.0.4 requires matplotlib>=2.1.0, which is not installed.`
Solution: Run `pip install matplotlib==3.5.2`.

- Problem: `fvcore 0.1.5.post20220512 requires tqdm, which is not installed.`
Solution: Run `pip install tqdm==4.64.0`.

Downgrade the following packages:
- `rapidfuzz` to version 2.1.1
- `numpy` to version 1.23.0
- `setuptools` to version 59.5.0
- `shapely` to version 1.8.2

## TRAINING ON A CUSTOM DATASET

To train a custom dataset, follow these steps:

1. Before proceeding, make sure you have completed steps 1 to 8 mentioned above.
2. Set up the directory and file hierarchy for the datasets as shown below:
```
AdelaiDet/datasets/coco
		/annotations
			/train2017
			/val2017
			/test2017
```

The `annotations` folder should contain the annotation JSON files for the train, val, and test data splits. The `train2017`, `val2017`, and `test2017` folders should contain the raw images for the respective datasets.

3. Run the following command to extract semantic labels from instance annotations:
python datasets/prepare_thing_sem_from_instance.py

After running this command, a directory named `thing2017` will be created inside the `coco` directory. This directory will contain `.npz` files of the images in the train dataset.

4. Edit `configs/BlendMask/Base-BlendMask.yaml` to configure the desired batch size, number of iterations, and learning rate.

5. If you are using just one GPU, edit adet/config/defaults.py and change the string SynBN to BN in C.MODEL.BASIS_MODULE.NORM = "BN" (line 141). File is also 	provided(defaults.py).

6. run the following commands to train your network:
	```
	!OMP_NUM_THREADS=1 python tools/train_net.py \      
    	--config-file configs/BlendMask/R_50_1x.yaml \
    	--num-gpus 1 \
    	OUTPUT_DIR training_dir/blendmask_R_50_1x
	```


NOTE: Pay attention to the names and paths of the training script, config file and directory wher the model will be saved, and the numberof GPU being used. 

`train_net.py` is the main training script. This script must be customized for the dataset. The datset must be registered and the registered name must be specicified in AdelaiDet/configs/BlendMask/Base-BlendMask.yaml.
	For simplicity the customized dataset has been provided 'train_net_vacc.py'. It can be renamed. 

	Using the original train_net.py from the original repo, returns: "AssertionError: Attribute 'thing_classes' in the metadata of 'coco_2017_train' cannot be set to a different value!..."




## POSSIBLE DEPENDENCY PROBLEMS AND SOLUTIONS
PROBLEM:site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module> LooseVersion = distutils.version.LooseVersion AttributeError: module 'distutils' has no attribute 'version'
SOLUTION: Downgrade `setuptools to 59.5.0`

PROBLEM:AttributeError: module 'numpy' has no attribute 'bool'. `np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. 
If you specifically wanted the numpy scalar type, use `np.bool_` here.
SOLUTION: Downgrade to `numpy 1.23.1`

PROBLEM: "Multi-polygon object not iterable"
SOLUTION: DOwngrade 'shapely' e.g.  `pip install shapely==1.8.2`
	
## SUPPLEMENTARY UTILITY FUNCTIONS

There is an utility python file that has functions to do tasks such as plot ground truth of images with annotations.The use of these supplemetary functions are illustrated in 'utilis_illustrate.ipynb' 

The functions description are as follows

1. plot_samples(dataset, number_of_samples) #This plots random ground truth and class labels of images in a registered dataset, 

2. plot_spec_anno_ (registered_dataset, img_name) #For a known specific image, in a dataset that has been registered, the ground truth annotations can be plotted usinf the function and the arguments

3. plot_groundt_anno (registered_dataset_name, dir_name) #To plot groud truth annotations of all images ina dataset and saved in the directory with name dir_nam
#dataset  must have been registered  with the name 'dataset_name'

4. The use of these supplemetary functions are illustrated in 'utilis_illustrate.ipynb'

NOTE to use these functions, it is helpful to:
1. have paths of json annotations and raw images assigned to string variables
2. register the dataset you maybe interested in
3. To use the prediction functions, in addition to 1. and 2, assign class definitions to a string variable

